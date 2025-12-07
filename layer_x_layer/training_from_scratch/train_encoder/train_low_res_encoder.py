# train_low_res_encoder.py
"""
Standalone training script for LowResContextEncoder.
Trains the encoder to extract meaningful features from low-resolution voxel grids
that can be used to condition higher-resolution generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    StateDictType,
    FullStateDictConfig,
)
from pathlib import Path
import numpy as np
import argparse
import os
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from functools import partial
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

class ResidualBlock3D(nn.Module):
    """Residual block for 3D convolutions with proper skip connections"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
        )
        
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()
        
        self.activation = nn.SiLU()
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.block(x)
        out = out + identity
        out = self.activation(out)
        return out


class LowResContextEncoder(nn.Module):
    """
    Enhanced encoder with better spatial and inter-layer awareness.
    """
    def __init__(self, in_channels=1, base_channels=64, mode='spatial_aware'):
        super().__init__()
        self.mode = mode
        
        self.encoder = nn.ModuleList([
            ResidualBlock3D(in_channels, base_channels, stride=1),
            ResidualBlock3D(base_channels, base_channels * 2, stride=2),
            ResidualBlock3D(base_channels * 2, base_channels * 4, stride=2),
        ])
        
        self.layer_pos_embedding = nn.Embedding(256, 64)
        self.pos_projection = nn.Linear(128 + 64, 128)
        
        if mode == 'spatial_aware':
            self.spatial_aggregator = nn.Sequential(
                nn.Conv2d(base_channels * 4, 256, 1),
                nn.GroupNorm(8, 256),
                nn.SiLU(),
                nn.Conv2d(256, 128, 1)
            )
            self.spatial_attention = nn.MultiheadAttention(
                embed_dim=128,
                num_heads=4,
                batch_first=True
            )
        elif mode == 'hierarchical':
            self.multiscale_fusion = nn.ModuleList([
                nn.Conv2d(base_channels, 32, 1),
                nn.Conv2d(base_channels * 2, 32, 1),
                nn.Conv2d(base_channels * 4, 64, 1),
            ])
            self.fusion_proj = nn.Linear(128, 128)
        else:
            self.layer_projection = nn.Sequential(
                nn.Linear(base_channels * 4, 256),
                nn.LayerNorm(256),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128)
            )
        
        self.inter_layer_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=512,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=2
        )
    
    def forward(self, low_res_voxels, target_granularity):
        B = low_res_voxels.shape[0]
        
        features_pyramid = []
        x = low_res_voxels
        
        for block in self.encoder:
            x = block(x)
            features_pyramid.append(x)
        
        features = features_pyramid[-1]
        layer_features = []
        
        for layer_idx in range(target_granularity):
            z_coord = (layer_idx + 0.5) / target_granularity * features.shape[-1]
            z_idx_low = int(z_coord)
            z_idx_high = min(z_idx_low + 1, features.shape[-1] - 1)
            alpha = z_coord - z_idx_low
            
            layer_feat = (1 - alpha) * features[:, :, :, :, z_idx_low] + \
                        alpha * features[:, :, :, :, z_idx_high]
            
            if self.mode == 'spatial_aware':
                layer_feat = self.spatial_aggregator(layer_feat)
                B_inner, C, H, W = layer_feat.shape
                layer_feat_flat = layer_feat.flatten(2).transpose(1, 2)
                
                layer_feat_attn, _ = self.spatial_attention(
                    layer_feat_flat, layer_feat_flat, layer_feat_flat
                )
                layer_feat = layer_feat_attn.mean(dim=1)
                
            elif self.mode == 'hierarchical':
                scale_features = []
                for j, (feat, proj) in enumerate(zip(features_pyramid, self.multiscale_fusion)):
                    z_idx = int(layer_idx * feat.shape[-1] / target_granularity)
                    z_idx = min(z_idx, feat.shape[-1] - 1)
                    
                    feat_slice = feat[:, :, :, :, z_idx]
                    feat_proj = proj(feat_slice)
                    feat_pool = F.adaptive_avg_pool2d(feat_proj, (1, 1)).flatten(1)
                    scale_features.append(feat_pool)
                
                layer_feat = torch.cat(scale_features, dim=1)
                layer_feat = self.fusion_proj(layer_feat)
                
            else:
                layer_feat = F.adaptive_avg_pool2d(layer_feat, (2, 2))
                layer_feat = layer_feat.flatten(1)
                layer_feat = self.layer_projection(layer_feat)
            
            layer_features.append(layer_feat)
        
        layer_features = torch.stack(layer_features, dim=1)
        
        layer_indices = torch.arange(target_granularity, device=layer_features.device)
        pos_emb = self.layer_pos_embedding(layer_indices).unsqueeze(0)
        
        layer_features = torch.cat([layer_features, pos_emb.expand(B, -1, -1)], dim=-1)
        layer_features = self.pos_projection(layer_features)
        layer_features = self.inter_layer_transformer(layer_features)
        
        return layer_features


class SimpleVoxelDecoder(nn.Module):
    """
    Simple decoder that reconstructs high-res voxels from encoded features.
    Used only during encoder training.
    """
    def __init__(self, feature_dim=128, target_granularity=32):
        super().__init__()
        self.target_granularity = target_granularity
        
        # Project features to spatial representation
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.SiLU()
        )
        
        # Reshape to 3D and upsample
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv3d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, layer_features):
        """
        Args:
            layer_features: [B, target_granularity, 128]
        Returns:
            voxels: [B, 1, H, W, D]
        """
        B, N, C = layer_features.shape
        
        # Project features
        features = self.feature_proj(layer_features)  # [B, N, 512]
        
        # Reshape to 3D volume (start small and upsample)
        # Initial spatial size depends on target granularity
        init_size = self.target_granularity // 8  # Will be upsampled 3x (2^3 = 8x)
        
        features = features.view(B, N, 512, 1, 1)  # [B, N, 512, 1, 1]
        features = features.permute(0, 2, 3, 4, 1)  # [B, 512, 1, 1, N]
        
        # Expand spatially
        features = F.interpolate(
            features, 
            size=(init_size, init_size, N),
            mode='trilinear',
            align_corners=False
        )
        
        # Decode to full resolution
        voxels = self.decoder(features)
        
        # Final resize to exact target size
        voxels = F.interpolate(
            voxels,
            size=(self.target_granularity, self.target_granularity, self.target_granularity),
            mode='trilinear',
            align_corners=False
        )
        
        return voxels


class PairedVoxelDataset(Dataset):
    """Dataset that loads paired low-res and high-res voxel grids"""
    def __init__(self, low_res_path, high_res_path, low_res_gran, high_res_gran, max_samples=None):
        self.low_res_path = Path(low_res_path)
        self.high_res_path = Path(high_res_path)
        self.low_res_gran = low_res_gran
        self.high_res_gran = high_res_gran
        
        # Get matching files
        low_res_files = {f.stem: f for f in self.low_res_path.glob('*.npy')}
        high_res_files = {f.stem: f for f in self.high_res_path.glob('*.npy')}
        
        # Find common files
        common_stems = set(low_res_files.keys()) & set(high_res_files.keys())
        
        if max_samples:
            common_stems = list(common_stems)[:max_samples]
        
        self.file_pairs = [
            (low_res_files[stem], high_res_files[stem]) 
            for stem in common_stems
        ]
        
        print(f"Found {len(self.file_pairs)} matching voxel pairs")
        print(f"Low-res: {low_res_gran}³, High-res: {high_res_gran}³")
    
    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        low_res_file, high_res_file = self.file_pairs[idx]
        
        low_res = np.load(low_res_file).astype(np.float32)
        high_res = np.load(high_res_file).astype(np.float32)
        
        # Add channel dimension
        low_res = torch.from_numpy(low_res).unsqueeze(0)  # [1, H, W, D]
        high_res = torch.from_numpy(high_res).unsqueeze(0)
        
        return low_res, high_res, low_res_file.stem


def parse_args():
    parser = argparse.ArgumentParser(description='Train LowResContextEncoder')
    
    # Data paths
    parser.add_argument('--low_res_path', type=str, required=True,
                        help='Path to low-resolution voxel directory')
    parser.add_argument('--high_res_path', type=str, required=True,
                        help='Path to high-resolution voxel directory')
    parser.add_argument('--low_res_gran', type=int, required=True,
                        help='Low-resolution granularity (e.g., 16)')
    parser.add_argument('--high_res_gran', type=int, required=True,
                        help='High-resolution granularity (e.g., 32)')
    
    # Training params
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split fraction')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples (for testing)')
    
    # Model params
    parser.add_argument('--encoder_mode', type=str, default='spatial_aware',
                        choices=['spatial_aware', 'hierarchical', 'simple'],
                        help='Encoder architecture mode')
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base channel count for encoder')
    
    # Distributed training
    parser.add_argument('--use_fsdp', action='store_true',
                        help='Use FSDP for distributed training')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./encoder_checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_interval', type=int, default=5,
                        help='Save checkpoint every N epochs')
    
    return parser.parse_args()


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' not in os.environ:
        return False, 0, 1, 0
    
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    
    return True, rank, world_size, local_rank


def get_fsdp_config(args):
    """Get FSDP configuration"""
    sharding_strategy = ShardingStrategy.FULL_SHARD
    
    if args.mixed_precision:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        mixed_precision = MixedPrecision(
            param_dtype=dtype,
            reduce_dtype=dtype,
            buffer_dtype=dtype,
        )
    else:
        mixed_precision = None
    
    auto_wrap_policy = partial(
        size_based_auto_wrap_policy,
        min_num_params=100_000
    )
    
    return {
        'sharding_strategy': sharding_strategy,
        'mixed_precision': mixed_precision,
        'auto_wrap_policy': auto_wrap_policy,
        'device_id': torch.cuda.current_device(),
        'sync_module_states': True,
    }


def train_epoch(encoder, decoder, dataloader, optimizer, device, epoch, rank, world_size):
    """Training loop for one epoch"""
    encoder.train()
    decoder.train()
    
    total_loss = 0
    total_mse = 0
    total_occupancy_diff = 0
    num_batches = 0
    
    for batch_idx, (low_res, high_res, stems) in enumerate(dataloader):
        low_res = low_res.to(device)
        high_res = high_res.to(device)
        
        optimizer.zero_grad()
        
        # Encode low-res features
        features = encoder(low_res, high_res.shape[-1])  # [B, N, 128]
        
        # Decode to high-res
        pred_high_res = decoder(features)  # [B, 1, H, W, D]
        
        # Compute losses
        mse_loss = F.mse_loss(pred_high_res, high_res)
        
        # Occupancy loss (encourage similar occupancy patterns)
        pred_occupancy = (pred_high_res > 0.5).float().mean()
        true_occupancy = (high_res > 0.5).float().mean()
        occupancy_loss = F.mse_loss(pred_occupancy, true_occupancy)
        
        # Feature diversity loss (prevent mode collapse)
        feature_std = features.std(dim=0).mean()
        diversity_loss = torch.exp(-feature_std)  # Penalize low diversity
        
        # Combined loss
        loss = mse_loss + 0.1 * occupancy_loss + 0.01 * diversity_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()),
            max_norm=1.0
        )
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        total_mse += mse_loss.item()
        total_occupancy_diff += abs(pred_occupancy.item() - true_occupancy.item())
        num_batches += 1
    
    # Synchronize metrics across ranks
    if world_size > 1:
        metrics = torch.tensor(
            [total_loss, total_mse, total_occupancy_diff, num_batches],
            device=device
        )
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_loss, total_mse, total_occupancy_diff, num_batches = metrics.tolist()
    
    avg_loss = total_loss / num_batches
    avg_mse = total_mse / num_batches
    avg_occ_diff = total_occupancy_diff / num_batches
    
    return {
        'loss': avg_loss,
        'mse': avg_mse,
        'occupancy_diff': avg_occ_diff
    }


def validate(encoder, decoder, dataloader, device, rank, world_size):
    """Validation loop"""
    encoder.eval()
    decoder.eval()
    
    total_loss = 0
    total_mse = 0
    num_batches = 0
    
    with torch.no_grad():
        for low_res, high_res, stems in dataloader:
            low_res = low_res.to(device)
            high_res = high_res.to(device)
            
            features = encoder(low_res, high_res.shape[-1])
            pred_high_res = decoder(features)
            
            mse_loss = F.mse_loss(pred_high_res, high_res)
            
            total_loss += mse_loss.item()
            total_mse += mse_loss.item()
            num_batches += 1
    
    if world_size > 1:
        metrics = torch.tensor([total_loss, total_mse, num_batches], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_loss, total_mse, num_batches = metrics.tolist()
    
    return {
        'val_loss': total_loss / num_batches,
        'val_mse': total_mse / num_batches
    }


def save_checkpoint(encoder, optimizer, epoch, metrics, output_path, rank, use_fsdp):
    """Save model checkpoint"""
    if use_fsdp:
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(encoder, StateDictType.FULL_STATE_DICT, save_policy):
            if rank == 0:
                state_dict = encoder.state_dict()
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metrics
                }, output_path)
                print(f"✓ Checkpoint saved: {output_path}")
    else:
        if rank == 0:
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics
            }, output_path)
            print(f"✓ Checkpoint saved: {output_path}")


def main():
    args = parse_args()
    
    # Setup distributed training
    is_distributed, rank, world_size, local_rank = setup_distributed()
    
    if is_distributed:
        device = torch.device(f'cuda:{local_rank}')
        print(f"[Rank {rank}/{world_size}] Running on GPU {local_rank}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Running on single device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    if rank == 0:
        print("\n" + "="*60)
        print("Loading Paired Voxel Dataset")
        print("="*60)
    
    full_dataset = PairedVoxelDataset(
        low_res_path=args.low_res_path,
        high_res_path=args.high_res_path,
        low_res_gran=args.low_res_gran,
        high_res_gran=args.high_res_gran,
        max_samples=args.max_samples
    )
    
    # Split into train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    if rank == 0:
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create models
    if rank == 0:
        print("\n" + "="*60)
        print("Initializing Models")
        print("="*60)
        print(f"Encoder mode: {args.encoder_mode}")
        print(f"Base channels: {args.base_channels}")
    
    encoder = LowResContextEncoder(
        in_channels=1,
        base_channels=args.base_channels,
        mode=args.encoder_mode
    ).to(device)
    
    decoder = SimpleVoxelDecoder(
        feature_dim=128,
        target_granularity=args.high_res_gran
    ).to(device)
    
    # Apply FSDP if requested
    if args.use_fsdp and is_distributed:
        fsdp_config = get_fsdp_config(args)
        encoder = FSDP(encoder, **fsdp_config)
        decoder = FSDP(decoder, **fsdp_config)
        if rank == 0:
            print("✓ FSDP enabled")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.learning_rate,
        weight_decay=1e-4
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # Training loop
    if rank == 0:
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
    
    metrics_history = {
        'train_loss': [],
        'train_mse': [],
        'val_loss': [],
        'val_mse': [],
        'epochs': []
    }
    
    best_val_loss = float('inf')
    
    # Create epoch-level progress bar (only on rank 0)
    epoch_pbar = tqdm(range(1, args.epochs + 1), desc="Training", disable=(rank != 0))
    
    for epoch in epoch_pbar:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Train
        train_metrics = train_epoch(
            encoder, decoder, train_loader, optimizer,
            device, epoch, rank, world_size
        )
        
        # Validate
        val_metrics = validate(
            encoder, decoder, val_loader, device, rank, world_size
        )
        
        scheduler.step()
        
        # Update progress bar with metrics
        if rank == 0:
            epoch_pbar.set_postfix({
                'loss': f'{train_metrics["loss"]:.4f}',
                'mse': f'{train_metrics["mse"]:.4f}',
                'val_loss': f'{val_metrics["val_loss"]:.4f}',
                'val_mse': f'{val_metrics["val_mse"]:.4f}'
            })
            
            # Save metrics
            metrics_history['train_loss'].append(train_metrics['loss'])
            metrics_history['train_mse'].append(train_metrics['mse'])
            metrics_history['val_loss'].append(val_metrics['val_loss'])
            metrics_history['val_mse'].append(val_metrics['val_mse'])
            metrics_history['epochs'].append(epoch)
        
        # Save checkpoint
        if epoch % args.checkpoint_interval == 0:
            checkpoint_path = output_dir / f'encoder_epoch_{epoch}.pth'
            save_checkpoint(
                encoder, optimizer, epoch, metrics_history,
                checkpoint_path, rank, args.use_fsdp
            )
        
        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            best_path = output_dir / 'encoder_best.pth'
            save_checkpoint(
                encoder, optimizer, epoch, metrics_history,
                best_path, rank, args.use_fsdp
            )
            if rank == 0:
                tqdm.write(f"  ✓ New best model! Val Loss: {best_val_loss:.4f}")
        
        if is_distributed:
            dist.barrier()
    
    # Final save
    final_path = output_dir / 'encoder_final.pth'
    save_checkpoint(
        encoder, optimizer, args.epochs, metrics_history,
        final_path, rank, args.use_fsdp
    )
    
    # Save metrics
    if rank == 0:
        with open(output_dir / 'training_metrics.json', 'w') as f:
            json.dump(metrics_history, f, indent=2)
        
        # Plot training curves
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(metrics_history['epochs'], metrics_history['train_loss'], label='Train')
        axes[0].plot(metrics_history['epochs'], metrics_history['val_loss'], label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(metrics_history['epochs'], metrics_history['train_mse'], label='Train')
        axes[1].plot(metrics_history['epochs'], metrics_history['val_mse'], label='Val')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MSE')
        axes[1].set_title('Reconstruction MSE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'training_curves.png', dpi=150)
        print("\n✓ Training curves saved")
    
    if rank == 0:
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Best Val Loss: {best_val_loss:.4f}")
        print(f"Checkpoints saved to: {output_dir}")
    
    # Cleanup
    if is_distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()