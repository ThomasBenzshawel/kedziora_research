# %%
from collections import OrderedDict, defaultdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union
import collections
import math
import random
import os
import json

import numpy as np
from numpy.random import RandomState

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import argparse
import os
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
    CPUOffload,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data.distributed import DistributedSampler
from functools import partial

import concurrent.futures
from pathlib import Path
import threading
from functools import lru_cache
import sqlite3

from difflib import get_close_matches
import logging

granularity = 16  # number of layers to generate and also the resolution of the unet input and output
def initialize_model_weights(model):
    """
    Initialize model weights properly before FSDP wrapping.
    Critical for avoiding cuDNN errors with FSDP + mixed precision.
    """
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            # Use kaiming for conv layers
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            # Xavier for linear layers with small gain for stability
            nn.init.xavier_uniform_(m.weight, gain=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm2d)):
            # Standard normalization layer init
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            # Small random values for embeddings
            nn.init.normal_(m.weight, mean=0, std=0.02)
    
    model.apply(init_weights)
    return model

def parse_args():
    """Parse command line arguments for training configuration"""
    parser = argparse.ArgumentParser(description='Train 3D Layer-by-Layer Diffusion Model')
    
    # Distributed training / sharding arguments
    parser.add_argument('--shard_model', action='store_true',
                        help='Enable FSDP automatic model sharding across all GPUs')
    parser.add_argument('--sharding_strategy', type=str, default='FULL_SHARD',
                        choices=['FULL_SHARD', 'SHARD_GRAD_OP', 'NO_SHARD'],
                        help='FSDP sharding strategy (default: FULL_SHARD for max memory savings)')
    parser.add_argument('--cpu_offload', action='store_true',
                        help='Offload parameters and gradients to CPU (saves GPU memory)')
    parser.add_argument('--mixed_precision', action='store_true', default=False,
                        help='Use mixed precision training (bfloat16 or float16)')
    
    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size (per GPU)')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--validation_interval', type=int, default=10,
                        help='Run validation every N epochs')
    parser.add_argument('--checkpoint_interval', type=int, default=50,
                        help='Save checkpoint every N epochs')
    
    return parser.parse_args()

# Parse arguments
args = parse_args()


def setup_distributed():
    """
    Initialize distributed training environment for FSDP.
    Returns: (is_distributed, rank, world_size, local_rank)
    """
    if not args.shard_model:
        return False, 0, 1, 0
    
    # Check if launched with torchrun/torch.distributed.launch
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        print("ERROR: --shard_model requires running with torchrun or torch.distributed.launch")
        print("Example: torchrun --nproc_per_node=auto train.py --shard_model")
        exit(1)
    
    # Get distributed training parameters from environment
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    # Set the device for this process
    torch.cuda.set_device(local_rank)
    
    # Initialize the process group
    dist.init_process_group(
        backend='nccl',  # Use NCCL for GPU communication
        init_method='env://',
    )
    
    return True, rank, world_size, local_rank


# Initialize distributed training
is_distributed, rank, world_size, local_rank = setup_distributed()

# Set device based on distributed mode
if is_distributed:
    device = torch.device(f"cuda:{local_rank}")
    print(f"[Rank {rank}/{world_size}] Running on GPU {local_rank}")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on single device: {device}")

# Configure CUDA
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

# Helper function to only print from main process
def print_main(*args, **kwargs):
    """Print only from rank 0 (main process)"""
    if not is_distributed or rank == 0:
        print(*args, **kwargs)

# note to self, layerxlayer is the conda env
# %%
def get_fsdp_config():
    """
    Get FSDP configuration based on command-line arguments.
    Returns a dictionary of FSDP parameters.
    """
    # Map string to ShardingStrategy enum
    strategy_map = {
        'FULL_SHARD': ShardingStrategy.FULL_SHARD,          # Shard params, grads, optimizer states
        'SHARD_GRAD_OP': ShardingStrategy.SHARD_GRAD_OP,    # Shard grads and optimizer states only
        'NO_SHARD': ShardingStrategy.NO_SHARD,              # DDP-like (for comparison)
    }
    sharding_strategy = strategy_map[args.sharding_strategy]
    
    # CPU offload configuration
    cpu_offload = CPUOffload(offload_params=True) if args.cpu_offload else None
    
    # Mixed precision configuration
    if args.mixed_precision:
        # Use bfloat16 if supported, otherwise float16
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        mixed_precision = MixedPrecision(
            param_dtype=dtype,
            reduce_dtype=dtype,
            buffer_dtype=dtype,
        )
    else:
        mixed_precision = None
    
    # Auto-wrap policy: automatically wrap modules with > 1M parameters
    # Adjust min_num_params based on your model size:
    #   - Larger values: less sharding, less communication overhead, higher memory
    #   - Smaller values: more sharding, more communication, lower memory per GPU
    auto_wrap_policy = partial(
        size_based_auto_wrap_policy,
        min_num_params=1_000_000  # 1M parameters threshold
    )
    
    config = {
        'sharding_strategy': sharding_strategy,
        'cpu_offload': cpu_offload,
        'mixed_precision': mixed_precision,
        'auto_wrap_policy': auto_wrap_policy,
        'backward_prefetch': BackwardPrefetch.BACKWARD_PRE,  # Prefetch next layer during backward
        'device_id': torch.cuda.current_device(),
        'sync_module_states': True,  # Sync module states during initialization
    }
    
    print_main("FSDP Configuration:")
    print_main(f"  Sharding Strategy: {args.sharding_strategy}")
    print_main(f"  CPU Offload: {args.cpu_offload}")
    print_main(f"  Mixed Precision: {args.mixed_precision}")
    print_main(f"  World Size: {world_size} GPUs")
    
    return config


# %%
# ========== T5 EMBEDDING CACHE ==========
class T5EmbeddingCacheSQL:
    """SQLite-based embedding cache for minimal memory footprint"""
    
    def __init__(self, cache_dir='./t5_cache', max_length=77, memory_cache_size=1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_length = max_length
        self.db_path = self.cache_dir / 'embeddings.db'
        self.memory_cache_size = memory_cache_size
        
        # Small LRU cache for frequently accessed embeddings
        self._memory_cache = {}
        self._cache_order = []
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                text_hash TEXT PRIMARY KEY,
                embedding BLOB
            )
        ''')
        conn.commit()
        conn.close()
    
    def load_cache(self):
        """
        Load cache from disk (no-op for SQLite since it's always persistent).
        This method exists for API compatibility with the original cache.
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM embeddings')
        count = cursor.fetchone()[0]
        conn.close()
        
        if count > 0:
            print(f"Found {count} cached embeddings in database")
        else:
            print("No existing cache found, will create new cache")
    
    def save_cache(self):
        """
        Save cache to disk (no-op for SQLite since writes are immediate).
        This method exists for API compatibility with the original cache.
        """
        # SQLite commits are immediate, so nothing to do
        # But we can count entries for informational purposes
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM embeddings')
        count = cursor.fetchone()[0]
        conn.close()
        print(f"Database contains {count} cached embeddings")
    
    def _get_cache_key(self, text):
        """Generate a unique key for a text description"""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()
    
    def get_embedding(self, text):
        """Get embedding from cache (memory -> disk)"""
        key = self._get_cache_key(text)
        
        # Check memory cache first
        if key in self._memory_cache:
            return self._memory_cache[key]
        
        # Check disk cache
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute('SELECT embedding FROM embeddings WHERE text_hash = ?', (key,))
        result = cursor.fetchone()
        conn.close()
        
        if result is not None:
            embedding = torch.from_numpy(
                np.frombuffer(result[0], dtype=np.float16).reshape(self.max_length, 768)
            )
            
            # Add to memory cache
            self._add_to_memory_cache(key, embedding)
            return embedding
        
        return None
    
    def _add_to_memory_cache(self, key, embedding):
        """Add to memory cache with LRU eviction"""
        if key in self._memory_cache:
            self._cache_order.remove(key)
        
        self._memory_cache[key] = embedding
        self._cache_order.append(key)
        
        # Evict oldest if cache is full
        if len(self._memory_cache) > self.memory_cache_size:
            oldest_key = self._cache_order.pop(0)
            del self._memory_cache[oldest_key]
    
    def add_embedding(self, text, embedding):
        """Add embedding to disk cache"""
        key = self._get_cache_key(text)
        embedding_bytes = embedding.cpu().numpy().astype(np.float16).tobytes()
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO embeddings (text_hash, embedding)
            VALUES (?, ?)
        ''', (key, embedding_bytes))
        conn.commit()
        conn.close()
        
        # Also add to memory cache
        self._add_to_memory_cache(key, embedding.cpu())
    
    def precompute_embeddings(self, dataloader_or_texts, tokenizer, model, device, batch_size=32):
        """
        Precompute embeddings and store in database (memory-efficient)
        
        Args:
            dataloader_or_texts: Either a DataLoader (from VoxelDataset) or a list of text strings
            tokenizer: T5 tokenizer
            model: T5 model
            device: torch device
            batch_size: Batch size for embedding computation (not DataLoader batch size)
        """
        from tqdm import tqdm
        import torch
        from torch.utils.data import DataLoader
        
        # Extract unique descriptions from DataLoader or use text list directly
        if isinstance(dataloader_or_texts, DataLoader):
            print("Extracting unique descriptions from DataLoader...")
            unique_descriptions = set()
            
            for batch in tqdm(dataloader_or_texts, desc="Scanning descriptions"):
                voxel_grids, descriptions = batch
                
                # Immediately discard voxel data to save memory
                del voxel_grids
                
                # Add descriptions to set (handles duplicates automatically)
                if isinstance(descriptions, (list, tuple)):
                    unique_descriptions.update(descriptions)
                else:
                    # Single description
                    unique_descriptions.add(descriptions)
                
                # Clear cache periodically
                if torch.cuda.is_available() and len(unique_descriptions) % 1000 == 0:
                    torch.cuda.empty_cache()
            
            texts = list(unique_descriptions)
            print(f"Found {len(texts)} unique descriptions from DataLoader")
        else:
            # Assume it's a list of texts
            texts = dataloader_or_texts
            print(f"Processing {len(texts)} descriptions from text list")
        
        # Find texts that aren't cached yet
        uncached_texts = []
        print("Checking cache for existing embeddings...")
        for t in tqdm(texts, desc="Cache check"):
            if self.get_embedding(t) is None:
                uncached_texts.append(t)
        
        if not uncached_texts:
            print("All embeddings already cached!")
            return
        
        print(f"Computing embeddings for {len(uncached_texts)} uncached texts...")
        model.eval()
        
        with torch.no_grad():
            for i in tqdm(range(0, len(uncached_texts), batch_size), desc="Caching embeddings"):
                batch_texts = uncached_texts[i:i+batch_size]
                
                # Tokenize
                text_inputs = tokenizer(
                    batch_texts,
                    padding='max_length',
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors='pt'
                ).to(device)
                
                # Compute embeddings
                embeddings = model(
                    input_ids=text_inputs.input_ids,
                    attention_mask=text_inputs.attention_mask
                ).last_hidden_state
                
                # Immediately process and store each embedding (moves to CPU in add_embedding)
                for text, emb in zip(batch_texts, embeddings):
                    self.add_embedding(text, emb)
                
                # Explicitly delete GPU tensors
                del text_inputs
                del embeddings
                
                # Clear CUDA cache every 10 batches to prevent fragmentation
                if torch.cuda.is_available() and (i // batch_size) % 10 == 0:
                    torch.cuda.empty_cache()
        
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.save_cache()  # Print count
        
    def get_batch_embeddings(self, texts, device):
        """Get batch of embeddings from cache"""
        embeddings = []
        for text in texts:
            emb = self.get_embedding(text)
            if emb is None:
                raise ValueError(f"Embedding not found in cache for text: {text[:50]}...")
            embeddings.append(emb)
        
        if is_distributed:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float16
        
        return torch.stack(embeddings).to(device, dtype=dtype)

# %%
# ========== VALIDATION METRICS ==========
class ValidationMetrics:
    """Track and compute validation metrics for 3D generation"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.losses = []
        self.layer_losses = []  # Per-layer losses
        self.voxel_occupancy = []  # How many voxels are occupied
        self.voxel_density = []  # Average voxel values
    
    def update(self, loss, layer_losses=None, generated_voxels=None):
        """Update metrics with new batch"""
        self.losses.append(loss)
        
        if layer_losses is not None:
            self.layer_losses.append(layer_losses)
        
        if generated_voxels is not None:
            # generated_voxels: [B, C, H, W, D]
            batch_size = generated_voxels.shape[0]
            for b in range(batch_size):
                voxels = generated_voxels[b].cpu().numpy()
                # Occupancy: percentage of voxels > threshold
                occupancy = (voxels > 0.5).mean()
                # Density: mean value of all voxels
                density = voxels.mean()
                
                self.voxel_occupancy.append(occupancy)
                self.voxel_density.append(density)
    
    def compute(self):
        """Compute average metrics"""
        metrics = {
            'val_loss': np.mean(self.losses) if self.losses else float('inf'),
            'val_occupancy': np.mean(self.voxel_occupancy) if self.voxel_occupancy else 0.0,
            'val_density': np.mean(self.voxel_density) if self.voxel_density else 0.0,
        }
        
        if self.layer_losses:
            # Average per-layer loss
            layer_losses_array = np.array(self.layer_losses)  # [num_batches, num_layers]
            metrics['val_layer_losses'] = layer_losses_array.mean(axis=0).tolist()
        
        return metrics
    
    def print_metrics(self, epoch):
        """Print metrics in a nice format"""
        metrics = self.compute()
        print(f"\n{'='*60}")
        print(f"Validation Metrics - Epoch {epoch}")
        print(f"{'='*60}")
        print(f"Loss:       {metrics['val_loss']:.4f}")
        print(f"Occupancy:  {metrics['val_occupancy']:.2%} (voxels > 0.5)")
        print(f"Density:    {metrics['val_density']:.4f} (mean voxel value)")
        print(f"{'='*60}\n")
        
        return metrics

# %%
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, channels, num_heads=8, dropout=0.2):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        num_groups = min(8, channels)
        while channels % num_groups != 0:
            num_groups -= 1
        self.norm = nn.GroupNorm(num_groups, channels)

        self.qkv = nn.Conv2d(channels, channels * 3, 1)  # 1x1 conv for Q, K, V
        self.proj = nn.Conv2d(channels, channels, 1)

        assert channels % num_heads == 0, f"channels {channels} must be divisible by num_heads {num_heads}"

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Normalize
        h = self.norm(x)
        
        # Get Q, K, V
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)  # [B, heads, HW, head_dim]
        k = k.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        v = v.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, heads, HW, head_dim]
        out = out.transpose(2, 3).contiguous().view(B, C, H, W)
        
        # Project and add residual
        out = self.proj(out)
        return x + out
    

# %%
class MultiHeadCrossAttention(nn.Module):
    """Updated to handle both vector and sequence context"""
    def __init__(self, channels, context_dim, num_heads=2):
        super().__init__()
        self.channels = channels
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.q = nn.Conv2d(channels, channels, 1)

        # Key and Value from context sequence
        self.to_kv = nn.Linear(context_dim, channels * 2)  # For both K and V
        
        self.proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x, context):
        """
        x: [B, C, H, W] - spatial features
        context: [B, seq_len, context_dim] - sequence of context vectors
        """
        B, C, H, W = x.shape
        
        # Normalize spatial features
        h = self.norm(x)
        
        # Query from spatial features
        q = self.q(h)  # [B, C, H, W]
        q = q.view(B, self.num_heads, self.head_dim, H * W)
        q = q.permute(0, 1, 3, 2)  # [B, heads, HW, head_dim]
        
        # Project context sequence to key and value
        seq_len = context.shape[1]
        kv = self.to_kv(context)  # [B, seq_len, channels * 2]
        kv = kv.view(B, seq_len, 2, self.num_heads, self.head_dim)
        k, v = kv[:, :, 0], kv[:, :, 1]  # Each is [B, seq_len, num_heads, head_dim]
        k = k.permute(0, 2, 1, 3)  # [B, num_heads, seq_len, head_dim]
        v = v.permute(0, 2, 1, 3)
        
        # Compute attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, heads, HW, seq_len]
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, heads, HW, head_dim]
        
        # Reshape back to spatial
        out = out.permute(0, 1, 3, 2).contiguous()
        out = out.view(B, C, H, W)
        
        # Output projection and residual
        out = self.proj(out)
        return x + out

# %%
class TransformerBlock(nn.Module):
    def __init__(self, channels, context_dim, num_heads=8, dropout=0.2):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(channels, num_heads, dropout)
        self.cross_attn = MultiHeadCrossAttention(channels, context_dim, num_heads)

        num_groups = min(8, channels)
        while channels % num_groups != 0:
            num_groups -= 1
    

        self.ffn = nn.Sequential(
            nn.GroupNorm(num_groups, channels),
            nn.Conv2d(channels, channels * 4, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(channels * 4, channels, 1),
            nn.Dropout(dropout)
        )


    def forward(self, x, context):
        x = self.self_attn(x)
        x = self.cross_attn(x, context)
        return x + self.ffn(x)

class AdaGN(nn.Module):
    def __init__(self, num_groups, num_channels, cond_channels):
        super().__init__()
        self.num_groups = num_groups
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.scale_shift = nn.Linear(cond_channels, num_channels * 2)
    
    def forward(self, x, cond):
        x = self.norm(x)
        scale, shift = self.scale_shift(cond).chunk(2, dim=-1)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


# Enhanced ResNet Block with optional attention and adaGN conditioning
class ResnetBlockWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, context_dim, 
                 layer_context_dim=64, use_attention=False, num_heads=8, num_groups=8):
        super().__init__()
        
        # Combined conditioning dimension (time + layer context)
        cond_dim = time_emb_dim + layer_context_dim
        
        # AdaGN layers that condition on BOTH time and layer
        self.norm1 = AdaGN(num_groups, in_channels, cond_dim)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.norm2 = AdaGN(num_groups, out_channels, cond_dim)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.activation = nn.SiLU()
        
        # Add transformer block if specified
        self.use_attention = use_attention
        if use_attention:
            self.attention = TransformerBlock(out_channels, context_dim, num_heads=num_heads)
        
        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, t, context_emb, layer_context=None):
        """
        x: [B, in_channels, H, W]
        t: [B, time_emb_dim] - time embedding
        context_emb: [B, seq_len, context_dim] - text/context embeddings for attention
        layer_context: [B, layer_context_dim] - layer positional embedding (optional)
        """
        # Combine time and layer context for conditioning
        if layer_context is not None:
            cond = torch.cat([t, layer_context], dim=-1)  # [B, time_emb_dim + layer_context_dim]
        else:
            # If no layer context, pad with zeros
            layer_zeros = torch.zeros(t.shape[0], self.norm1.scale_shift.in_features - t.shape[1], 
                                     device=t.device, dtype=t.dtype)
            cond = torch.cat([t, layer_zeros], dim=-1)
        
        # First ResNet block: norm -> activation -> conv
        h = self.norm1(x, cond)
        h = self.activation(h)
        h = self.conv1(h)
        
        # Second ResNet block: norm -> activation -> conv
        h = self.norm2(h, cond)
        h = self.activation(h)
        h = self.conv2(h)
        
        # Apply attention if enabled
        if self.use_attention:
            h = self.attention(h, context_emb)
        
        # Add skip connection
        return h + self.shortcut(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        dtype = x.dtype  
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0,1,0,0))
        return emb

class ExponentialMovingAverage:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        # Backup current parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# %%
# forward diffusion that creates a corrupted input

class ForwardDiffusion():
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, schedule='cosine'):
        """
        Initialize forward diffusion process with configurable beta schedule.
        
        Args:
            timesteps: Number of diffusion timesteps
            beta_start: Starting beta value (for linear schedule)
            beta_end: Ending beta value (for linear schedule)
            schedule: 'linear' or 'cosine' (cosine is generally better)
        """
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule = schedule
        
        # Create beta schedule
        if schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(timesteps)
        elif schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}. Use 'linear' or 'cosine'")
        
        self.alphas = 1.0 - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim=0)

        # Precompute sqrt_alpha_hats and sqrt_one_minus_alpha_hats for efficiency
        self.sqrt_alpha_hats = torch.sqrt(self.alpha_hats)
        self.sqrt_one_minus_alpha_hats = torch.sqrt(1 - self.alpha_hats)

        # ensure all tensors are on the same device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_hats = self.alpha_hats.to(device)
        self.sqrt_alpha_hats = self.sqrt_alpha_hats.to(device)
        self.sqrt_one_minus_alpha_hats = self.sqrt_one_minus_alpha_hats.to(device)
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        
        This schedule maintains more signal in the early timesteps and is often
        better for high-resolution images/voxels.
        
        Args:
            timesteps: Number of diffusion steps
            s: Small offset to prevent beta from being too small near t=0
        """
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps)
        
        # Cosine schedule for alpha_hat
        alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        
        # Calculate betas from alpha_cumprod
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        
        # Clip betas to prevent numerical instability
        # Standard practice is to clip between 0.0001 and 0.9999
        betas = torch.clip(betas, 0.0001, 0.9999)
        
        return betas

    def forward(self, x0, t, noise=None, dtype=torch.float32):
        if noise is None:
            if is_distributed:
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                dtype = torch.float16
            noise = torch.randn_like(x0).to(dtype=dtype)

        sqrt_alpha_hat = self.sqrt_alpha_hats[t].view(-1, 1, 1, 1).to(x0.device)
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hats[t].view(-1, 1, 1, 1).to(x0.device)
        
        xt = sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * noise
        return xt, noise

    def get_variance_schedule(self, t):
        # Return the variance at timestep t (posterior variance)
        if t == 0:
            return self.betas[0]
        else:
            alpha_hat_t = self.alpha_hats[t]
            alpha_hat_t_minus_1 = self.alpha_hats[t - 1]
            return self.betas[t] * (1 - alpha_hat_t_minus_1) / (1 - alpha_hat_t)

class RelativePositionBias(nn.Module):
    def __init__(self, num_heads, max_distance=32):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        # +1 for exactly max_distance away
        self.embeddings = nn.Embedding(2 * max_distance + 1, num_heads)
    
    def forward(self, N):
        """
        N: number of previous layers
        Returns: [num_heads, N, N] bias matrix
        """
        device = self.embeddings.weight.device
        positions = torch.arange(N, device=device)
        
        # Relative positions: how far is layer j from layer i
        relative_positions = positions[None, :] - positions[:, None]  # [N, N]
        relative_positions = torch.clamp(
            relative_positions, 
            -self.max_distance, 
            self.max_distance
        )
        relative_positions = relative_positions + self.max_distance
        
        # Get embeddings and transpose to [num_heads, N, N]
        bias = self.embeddings(relative_positions)  # [N, N, num_heads]
        return bias.permute(2, 0, 1)  # [num_heads, N, N]


class MultiLayerCrossAttentionBlock(nn.Module):
    """
    Cross-attention block that attends from current layer to ALL previous layers.
    """
    def __init__(self, channels, layer_context_dim, num_heads=8, dropout=0.0, max_distance=32):
        super().__init__()
        self.channels = channels
        self.layer_context_dim = layer_context_dim
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, f"channels {channels} must be divisible by num_heads {num_heads}"
        
        # Normalization
        num_groups = min(8, channels)
        while channels % num_groups != 0:
            num_groups -= 1
        self.norm_q = nn.GroupNorm(num_groups, channels)
        self.norm_kv = nn.GroupNorm(num_groups, channels)
        
        # Projections
        self.to_q = nn.Conv2d(channels, channels, 1)
        self.to_k = nn.Conv2d(channels, channels, 1)
        self.to_v = nn.Conv2d(channels, channels, 1)
        
        # Layer positional encoding projection
        self.layer_pos_proj = nn.Linear(layer_context_dim, channels)
        
        # Relative position bias for layer-to-layer attention
        self.relative_position_bias = RelativePositionBias(
            num_heads=num_heads,
            max_distance=max_distance
        )
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )
        
    def forward(self, x, prev_layers, layer_positions, current_layer_idx=None):
        """
        x: [B, C, H, W] - current layer features
        prev_layers: [B, N, C, H, W] - all previous layer features
        layer_positions: [B, N, layer_context_dim] - positional encodings
        current_layer_idx: Tensor [B] - index of current layer (for relative bias)
        """
        B, C, H, W = x.shape
        N = prev_layers.shape[1] if prev_layers is not None and prev_layers.numel() > 0 else 0
        
        if N == 0:
            return x
        
        # Get query from current layer
        q = self.norm_q(x)
        q = self.to_q(q)
        q = q.view(B, self.num_heads, self.head_dim, H * W)
        q = q.permute(0, 1, 3, 2)  # [B, num_heads, HW, head_dim]
        
        # Process all previous layers
        prev_layers_flat = prev_layers.view(B * N, C, H, W)
        prev_layers_norm = self.norm_kv(prev_layers_flat)
        
        k = self.to_k(prev_layers_norm)
        v = self.to_v(prev_layers_norm)
        
        # Reshape for multi-head attention
        k = k.view(B, N, self.num_heads, self.head_dim, H * W)
        k = k.permute(0, 2, 1, 4, 3).reshape(B, self.num_heads, N * H * W, self.head_dim)
        
        v = v.view(B, N, self.num_heads, self.head_dim, H * W)
        v = v.permute(0, 2, 1, 4, 3).reshape(B, self.num_heads, N * H * W, self.head_dim)
        
        # Add positional encodings
        if layer_positions is not None:
            pos_emb = self.layer_pos_proj(layer_positions)
            pos_emb = pos_emb.view(B, N, self.num_heads, self.head_dim)
            pos_emb = pos_emb.permute(0, 2, 1, 3)
            pos_emb = pos_emb.unsqueeze(3).expand(-1, -1, -1, H * W, -1)
            pos_emb = pos_emb.reshape(B, self.num_heads, N * H * W, self.head_dim)
            k = k + pos_emb
        
        # Compute attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, num_heads, HW, N*HW]
        
        # Add relative position bias
        if current_layer_idx is not None:
            device = attn.device
            
            # Previous layer indices: [0, 1, 2, ..., N-1]
            prev_indices = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)  # [B, N]
            
            # Compute relative distances from current layer to each previous layer
            relative_distances = current_layer_idx.unsqueeze(1) - prev_indices  # [B, N]
            relative_distances = torch.clamp(
                relative_distances, 
                -self.relative_position_bias.max_distance, 
                self.relative_position_bias.max_distance
            )
            relative_distances = relative_distances + self.relative_position_bias.max_distance
            
            # Get bias: [B, N, num_heads]
            bias = self.relative_position_bias.embeddings(relative_distances)
            bias = bias.permute(0, 2, 1)  # [B, num_heads, N]
            
            # Reshape bias to match attention shape: [B, num_heads, HW, N*HW]
            # Each spatial position gets the same layer-level bias
            bias = bias.unsqueeze(2)  # [B, num_heads, 1, N]
            bias = bias.repeat(1, 1, H * W, 1)  # [B, num_heads, HW, N]
            bias = bias.unsqueeze(-1).expand(-1, -1, -1, -1, H * W)  # [B, num_heads, HW, N, HW]
            bias = bias.reshape(B, self.num_heads, H * W, N * H * W)  # [B, num_heads, HW, N*HW]
            
            # Add bias to attention scores
            attn = attn + bias
        
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        
        # Output projection and residual
        out = self.to_out(out)
        return x + out


class LayerXLayerDiffusionModelV2(nn.Module):
    """
    Enhanced version with cross-attention to ALL previous layers.
    Drop-in replacement for LayerXLayerDiffusionModel.
    """
    def __init__(self, layer_context_dim=64, granularity=128,
                 text_context_dim=768, max_context_layers=None, in_channels=1, model_channels=128, context_dim=512,
                 attention_resolutions=[8, 16]):
        """
        Args:
            layer_context_dim: Dimension of layer context embeddings
            granularity: Number of layers in 3D volume
            text_context_dim: Dimension of text embeddings (768 for T5-base)
            max_context_layers: Maximum number of previous layers to attend to
                               None = all layers (high memory)
                               16-32 = recommended for balance
        """
        super().__init__()
        self.layer_context_dim = layer_context_dim
        self.granularity = granularity
        self.text_context_dim = text_context_dim
        self.max_context_layers = max_context_layers
        self.model_channels = model_channels
        self.attention_resolutions = attention_resolutions
        
        # Positional encoding for layer indices
        self.layer_pos_emb = SinusoidalPosEmb(layer_context_dim)
        
        # Layer context conditioning
        self.layer_context_conditioning_mlp = nn.Sequential(
            nn.Linear(layer_context_dim + text_context_dim, layer_context_dim * 2),
            nn.GELU(),
            nn.Linear(layer_context_dim * 2, text_context_dim)
        )
        
        # Multi-layer cross-attention
        self.multi_layer_cross_attn = MultiLayerCrossAttentionBlock(
            channels=self.model_channels,
            layer_context_dim=layer_context_dim,
            num_heads=8,
            max_distance=32  # Can adjust this based on your typical layer distances
        )
        
        # Encoder for previous layers
        self.prev_layer_encoder = nn.Sequential(
            nn.Conv2d(1, self.model_channels // 2, 3, padding=1),
            nn.GroupNorm(8, self.model_channels // 2),
            nn.SiLU(),
            nn.Conv2d(self.model_channels // 2, self.model_channels, 3, padding=1),
        )
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Input
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Down blocks with attention at specific resolutions
        # For 32x32 input: 32 -> 16 -> 8
        self.down_block1 = ResnetBlockWithAttention(
            model_channels, model_channels * 2, time_embed_dim, context_dim,
            use_attention=(32 in attention_resolutions)
        )
        self.down_block2 = ResnetBlockWithAttention(
            model_channels * 2, model_channels * 4, time_embed_dim, context_dim,
            use_attention=(16 in attention_resolutions)
        )
        self.down_block3 = ResnetBlockWithAttention(
            model_channels * 4, model_channels * 4, time_embed_dim, context_dim,
            use_attention=(8 in attention_resolutions)
        )
        
        self.downsample1 = nn.MaxPool2d(2)
        self.downsample2 = nn.MaxPool2d(2)
        
        # Middle block with attention (always at lowest resolution)
        self.mid_block = ResnetBlockWithAttention(
            model_channels * 4, model_channels * 4, time_embed_dim, context_dim,
            use_attention=True
        )
        
        # Up blocks with attention
        self.up_block1 = ResnetBlockWithAttention(
            model_channels * 4, model_channels * 4, time_embed_dim, context_dim,
            use_attention=(8 in attention_resolutions)
        )
        self.up_block2 = ResnetBlockWithAttention(
            model_channels * 4 + model_channels * 4, model_channels * 2, time_embed_dim, context_dim,
            use_attention=(16 in attention_resolutions)
        )
        self.up_block3 = ResnetBlockWithAttention(
            model_channels * 2 + model_channels * 2, model_channels, time_embed_dim, context_dim,
            use_attention=(32 in attention_resolutions)
        )
        
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Output
        self.output_conv = nn.Conv2d(model_channels, in_channels, 3, padding=1)

    def forward(self, x, t, l, context, prev_layers=None):
        """
        Forward pass with cross-attention to all previous layers.
        
        Args:
            x: [B, C, H, W] - noisy input for current layer
            t: [B] - diffusion timestep
            l: [B] - layer index (0 to granularity-1)
            context: [B, text_context_dim] - text embeddings
            prev_layers: [B, N, C, H, W] or None - ALL previous clean layers
        
        Returns:
            predicted_noise: [B, C, H, W]
        """
        B = x.shape[0]

        # get model parameter dtype
        model_dtype = next(self.parameters()).dtype             
        # Create layer position embedding

        l = l.to(model_dtype)
        layer_pos = self.layer_pos_emb(l)  # [B, layer_context_dim]
        # Ensure layer_pos is in the same dtype as the model
        layer_pos = layer_pos.to(model_dtype)
        context = context.to(model_dtype)
        t = t.to(model_dtype)
        x = x.to(model_dtype)
        prev_layers = prev_layers.to(model_dtype) if prev_layers is not None else None

        # reshape the layer position to be [B, 1, layer_context_dim]
        layer_pos_expanded = layer_pos.unsqueeze(1)  # [B, 1, layer_context_dim]

        # Expand layer_pos to match context's sequence length
        seq_len = context.shape[1]  
        layer_pos_expanded = layer_pos_expanded.expand(-1, seq_len, -1)  # [B, sequence, layer_context_dim]

        # Combine layer position and text context
        combined_context = torch.cat([layer_pos_expanded, context], dim=-1)  # [B, sequence, layer_context_dim + text_context_dim]
        # Process combined context through MLP
        context = self.layer_context_conditioning_mlp(combined_context)  # [B, sequence, text_context_dim]

        # Get features through input conv
        h = self.input_conv(x)

        # Process previous layers if they exist
        if prev_layers is not None and prev_layers.numel() > 0:
            # Apply sliding window if specified
            if self.max_context_layers is not None and prev_layers.shape[1] > self.max_context_layers:
                prev_layers = prev_layers[:, -self.max_context_layers:]
            
            N = prev_layers.shape[1]
            
            # Encode previous layers
            prev_layers_flat = prev_layers.view(B * N, prev_layers.shape[2], 
                                               prev_layers.shape[3], prev_layers.shape[4])
            prev_features = self.prev_layer_encoder(prev_layers_flat)
            prev_features = prev_features.view(B, N, prev_features.shape[1], 
                                              prev_features.shape[2], prev_features.shape[3])
            
            # Create positional encodings for all previous layers
            layer_indices = torch.arange(N, device=x.device, dtype=model_dtype).unsqueeze(0).expand(B, -1)  # [B, N]
            layer_indices_flat = layer_indices.reshape(-1)  # [B*N] - flatten to 1D
            layer_positions = self.layer_pos_emb(layer_indices_flat)  # [B*N, layer_context_dim]
            layer_positions = layer_positions.view(B, N, -1)  # [B, N, layer_context_dim] - reshape back
        
            # Convert l to long for indexing
            current_layer_idx = l.long()
            
            # Apply multi-layer cross-attention with relative position bias
            h = self.multi_layer_cross_attn(h, prev_features, layer_positions, current_layer_idx)
            
        # Continue through UNet
        time_emb = self.time_embed(t)
        h1 = self.down_block1(h, time_emb, context)
        h = self.downsample1(h1)
        h2 = self.down_block2(h, time_emb, context)
        h = self.downsample2(h2)
        h3 = self.down_block3(h, time_emb, context)
        h = self.mid_block(h3, time_emb, context)
        h = self.up_block1(h, time_emb, context)
        h = self.upsample1(h)
        h = torch.cat([h, h2], dim=1)
        h = self.up_block2(h, time_emb, context)
        h = self.upsample2(h)
        h = torch.cat([h, h1], dim=1)
        h = self.up_block3(h, time_emb, context)
        
        return self.output_conv(h)
            

# %% [markdown]
# # create the 3d voxel dataset 
# (inside the folder_path_{granularity}, there are .npy files of shape (granularity, granularity, granularity) with values in [0, 1])
# the user will make a voxel dataset object for each granularity they want to train on

# %%

class VoxelDataset(Dataset):
    def __init__(self, npy_folder_path, description_folder_path, transform=None, granularity=128, test_count=0):
        self.npy_folder_path = Path(npy_folder_path)
        self.description_folder_path = Path(description_folder_path)
        self.transform = transform
        self.granularity = granularity
        self.default_description = "A 3D object"
        
        # Track description misses for debugging
        self.description_misses = {}  
        
        print("Loading file lists...")
        
        # Only store file paths (minimal memory)
        if test_count is not None and test_count > 0:
            file_list = []
            for entry in os.scandir(npy_folder_path):
                if entry.name.endswith('.npy') and entry.is_file():
                    file_list.append(self.npy_folder_path / entry.name)
                    if len(file_list) >= test_count:
                        break
            print(f"TEST MODE: Using only {len(file_list)} samples")
        else:
            file_list = [
                self.npy_folder_path / entry.name
                for entry in os.scandir(npy_folder_path)
                if entry.name.endswith('.npy') and entry.is_file()
            ]
        
        self.file_list = file_list
        
        # Cache available description filenames for matching
        self.available_descriptions = set()
        for entry in os.scandir(description_folder_path):
            if entry.name.endswith('.txt') and entry.is_file():
                self.available_descriptions.add(entry.name[:-4])  # Remove .txt extension
        
        print(f"Found {len(self.file_list)} .npy files")
        print(f"Found {len(self.available_descriptions)} description files")
        print(f"Voxel grid granularity: {self.granularity}x{self.granularity}x{self.granularity}")
    
    @lru_cache(maxsize=10000)  # Cache recently accessed descriptions
    def _load_description(self, filename_stem):
        """Load a single description file on-demand with caching"""
        desc_file = self.description_folder_path / f"{filename_stem}.txt"
        
        if desc_file.exists():
            try:
                with open(desc_file, 'r') as f:
                    return f.read().strip()
            except Exception as e:
                print(f"Error loading {desc_file}: {e}")
                return self.default_description
        else:
            # Log the miss with most likely match
            closest_matches = get_close_matches(filename_stem, self.available_descriptions, n=1, cutoff=0.6)
            closest_match = closest_matches[0] if closest_matches else "No close match found"
            
            miss_info = {
                'missing': filename_stem,
                'closest_match': closest_match
            }
            self.description_misses.append(miss_info)
            
            return self.default_description
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # Load voxel data (already on-demand)
        voxel_grid = np.load(self.file_list[idx]).astype(np.float16)
        assert voxel_grid.shape == (self.granularity, self.granularity, self.granularity)
        
        if self.transform:
            voxel_grid = self.transform(voxel_grid)
        
        voxel_grid = torch.from_numpy(voxel_grid).unsqueeze(0)
        
        # Load description on-demand (cached)
        filename_key = self.file_list[idx].stem
        description = self._load_description(filename_key)
        
        return voxel_grid, description
    
    def save_miss_report(self, output_path="description_misses.txt"):
        """Save a report of all description misses"""
        if not self.description_misses:
            print("No description misses to report!")
            return
        
        with open(output_path, 'w') as f:
            f.write(f"Description Misses Report\n")
            f.write(f"Total misses: {len(self.description_misses)}\n")
            f.write(f"{'='*80}\n\n")
            
            for miss in self.description_misses:
                f.write(f"Missing: {miss['missing']}\n")
                f.write(f"Closest match: {miss['closest_match']}\n")
                f.write(f"{'-'*80}\n")
        
        print(f"Miss report saved to {output_path}")
        print(f"Total misses: {len(self.description_misses)}")


# %%
# create a 3d difussion trainer for the layer x layer model

class LayerXLayerDiffusionTrainerV2:
    """
    Enhanced trainer that accumulates ALL previous layers.
    Drop-in replacement for LayerXLayerDiffusionTrainer.
    """
    def __init__(self, model: LayerXLayerDiffusionModelV2, diffusion, scheduler, layer_by_layer_convergence=True, 
                 teacher_forcing=True, use_ddim=True, ddim_steps=50, ddim_eta=0.0):
        self.model = model
        self.diffusion = diffusion
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        self.scheduler = scheduler
        self.layer_by_layer_convergence = layer_by_layer_convergence
        self.teacher_forcing = teacher_forcing
        self.use_ddim = use_ddim
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta

    def train_step(self, x0, context):
        """
        Training step with cross-attention to all previous layers.
        
        Args:
            x0: [B, C, H, W, D] - full 3D object
            context: [B, context_dim] - text embeddings
        
        Returns:
            avg_loss: Average loss across all layers
            None: (for compatibility with original interface)
        """
        self.model.train()
        self.optimizer.zero_grad()

        batch_size = x0.size(0)
        device = x0.device
        total_loss = 0
        dtype = next(self.model.parameters()).dtype

        # Accumulate all previous layers
        prev_layers_list = []

        for layer_idx in range(self.model.granularity):
            # Extract current layer
            current_layer = x0[:, :, :, :, layer_idx]  # [B, C, H, W]

            # Sample timestep
            t = torch.randint(0, self.diffusion.timesteps, (batch_size,), device=device).long()

            # Add noise
            xt, noise = self.diffusion.forward(current_layer, t)
            
            # Layer index tensor
            l = torch.full((batch_size,), layer_idx, device=device, dtype=dtype)

            # Stack all previous layers
            if len(prev_layers_list) > 0:
                prev_layers = torch.stack(prev_layers_list, dim=1)  # [B, N, C, H, W]
            else:
                prev_layers = None

            # Predict noise with cross-attention to ALL previous layers
            predicted_noise = self.model(xt, t, l, context, prev_layers)

            # Compute loss
            loss = F.mse_loss(predicted_noise.float(), noise.float())
            total_loss += loss.detach()
            loss.backward()

            # Add current layer to history
            if self.teacher_forcing:
                prev_layers_list.append(current_layer.detach())
            else:
                # Use denoised prediction
                alpha_hat_t = self.diffusion.alpha_hats[t].view(-1, 1, 1, 1)
                denoised = (xt - torch.sqrt(1 - alpha_hat_t) * predicted_noise) / torch.sqrt(alpha_hat_t)
                prev_layers_list.append(denoised.detach())

        # Optimize
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        avg_loss = total_loss / self.model.granularity
        return avg_loss, None

    def sample(self, context, shape, device):
        """Unified sampling method"""
        if self.use_ddim:
            return self.ddim_layer_convergence_sample(context, shape, device, 
                                                     self.ddim_steps, self.ddim_eta)
        else:
            return self.layer_convergence_sample(context, shape, device)

    def layer_convergence_sample(self, context, shape, device):
        """
        Generate samples with layer-by-layer convergence and full cross-attention.
        """
        self.model.eval()
        # set up master dtype
        model_dtype = next(self.model.parameters()).dtype
        with torch.no_grad():
            batch_size = shape[0]
            voxel_grid = torch.zeros((batch_size, shape[1], shape[2], shape[3], 
                                     self.model.granularity), device=device, dtype=model_dtype)

            # Ensure all tensors are in the correct dtype
            voxel_grid = voxel_grid.to(dtype=model_dtype)

            generated_layers = []

            for layer_idx in range(self.model.granularity):
                l = torch.full((batch_size,), layer_idx, device=device, dtype=model_dtype)
                x = torch.randn(shape, device=device, dtype=model_dtype)

                # Stack all previously generated layers
                if len(generated_layers) > 0:
                    prev_layers = torch.stack(generated_layers, dim=1)
                else:
                    prev_layers = None

                # Run diffusion to convergence
                for t in reversed(range(self.diffusion.timesteps)):
                    t_batch = torch.full((batch_size,), t, device=device, dtype=model_dtype)
                    predicted_noise = self.model(x, t_batch, l, context, prev_layers)

                    alpha_t = self.diffusion.alphas[t]
                    alpha_hat_t = self.diffusion.alpha_hats[t]
                    beta_t = self.diffusion.betas[t]

                    if t > 0:
                        noise = torch.randn_like(x)
                    else:
                        noise = torch.zeros_like(x)

                    x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * predicted_noise) + torch.sqrt(beta_t) * noise

                voxel_grid[:, :, :, :, layer_idx] = x
                generated_layers.append(x.detach())

            return voxel_grid.clamp(0, 1)

    def ddim_layer_convergence_sample(self, context, shape, device, ddim_steps=50, eta=0.0):
        """
        DDIM sampling with full cross-attention.
        """
        self.model.eval()
        model_dtype = next(self.model.parameters()).dtype
        with torch.no_grad():
            batch_size = shape[0]
            voxel_grid = torch.zeros((batch_size, shape[1], shape[2], shape[3], 
                                     self.model.granularity), device=device, dtype=model_dtype)

            # Create DDIM timesteps
            timestep_interval = self.diffusion.timesteps // ddim_steps
            ddim_timesteps = list(range(0, self.diffusion.timesteps, timestep_interval))
            ddim_timesteps.reverse()
            
            generated_layers = []

            for layer_idx in range(self.model.granularity):
                l = torch.full((batch_size,), layer_idx, device=device, dtype=model_dtype)
                x = torch.randn(shape, device=device, dtype=model_dtype)

                # Stack previous layers
                if len(generated_layers) > 0:
                    prev_layers = torch.stack(generated_layers, dim=1)
                else:
                    prev_layers = None
                
                for i, t in enumerate(ddim_timesteps):
                    t_batch = torch.full((batch_size,), t, device=device, dtype=model_dtype)
                    predicted_noise = self.model(x, t_batch, l, context, prev_layers)
                    
                    # DDIM update rule
                    alpha_hat_t = self.diffusion.alpha_hats[t]
                    pred_x0 = (x - torch.sqrt(1 - alpha_hat_t) * predicted_noise) / torch.sqrt(alpha_hat_t)
                    pred_x0 = pred_x0.clamp(-1, 1)
                    
                    if i < len(ddim_timesteps) - 1:
                        t_prev = ddim_timesteps[i + 1]
                        alpha_hat_t_prev = self.diffusion.alpha_hats[t_prev]
                        dir_xt = torch.sqrt(1 - alpha_hat_t_prev - eta**2) * predicted_noise
                        
                        if eta > 0:
                            noise = torch.randn_like(x) * eta
                        else:
                            noise = 0
                        
                        x = torch.sqrt(alpha_hat_t_prev) * pred_x0 + dir_xt + noise
                    else:
                        x = pred_x0
                
                voxel_grid[:, :, :, :, layer_idx] = x
                generated_layers.append(x.detach())
            
            return voxel_grid.clamp(0, 1)
    
    def validate(self, dataloader, context_cache, device, max_batches=None):
        """
        Validation loop with full cross-attention.
        """
        self.model.eval()
        total_loss = 0
        total_occupancy = 0
        total_density = 0
        num_batches = 0
        model_dtype = next(self.model.parameters()).dtype
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                
                x0 = batch['voxel'].to(device)
                text_descriptions = batch['description']
                context = context_cache.get_embeddings(text_descriptions).to(device)
                
                batch_size = x0.size(0)
                batch_loss = 0
                prev_layers_list = []
                
                for layer_idx in range(self.model.granularity):
                    current_layer = x0[:, :, :, :, layer_idx]
                    t = torch.randint(0, self.diffusion.timesteps, (batch_size,), device=device).long()
                    xt, noise = self.diffusion.forward(current_layer, t)
                    l = torch.full((batch_size,), layer_idx, device=device, dtype=model_dtype)
                    
                    # Stack previous layers
                    if len(prev_layers_list) > 0:
                        prev_layers = torch.stack(prev_layers_list, dim=1)
                    else:
                        prev_layers = None
                    
                    predicted_noise = self.model(xt, t, l, context, prev_layers)
                    
                    if self.teacher_forcing:
                        prev_layers_list.append(current_layer.detach())
                    else:
                        alpha_hat_t = self.diffusion.alpha_hats[t].view(-1, 1, 1, 1)
                        denoised = (xt - torch.sqrt(1 - alpha_hat_t) * predicted_noise) / torch.sqrt(alpha_hat_t)
                        prev_layers_list.append(denoised.detach())
                    
                    loss = F.mse_loss(predicted_noise, noise)
                    batch_loss += loss.item()
                
                total_loss += batch_loss / self.model.granularity
                
                # Calculate occupancy and density metrics
                threshold = 0.5
                occupancy = (x0 > threshold).float().mean().item()
                density = x0.mean().item()
                total_occupancy += occupancy
                total_density += density
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_occupancy = total_occupancy / num_batches if num_batches > 0 else 0
        avg_density = total_density / num_batches if num_batches > 0 else 0
        
        return {
            'val_loss': avg_loss,
            'val_occupancy': avg_occupancy,
            'val_density': avg_density
        }


class TrainingVoxelVisualizer:
    def __init__(self, save_dir, max_voxels=30000):
        """
        Visualizer for generating and displaying voxel samples during training
        
        Args:
            save_dir: Directory to save visualizations
            max_voxels: Maximum number of voxels to display (for performance)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_voxels = max_voxels
    
    def visualize_single_voxel(self, voxel_grid, ax, threshold=0.5, view_angle=(30, 45), colormap='viridis'):
        """
        Visualize a single voxel grid in 3D
        
        Args:
            voxel_grid: [H, W, D] numpy array
            ax: matplotlib 3D axis
            threshold: Voxel occupancy threshold
            view_angle: (elevation, azimuth) tuple for viewing angle
            colormap: Matplotlib colormap to use
        """
        # Create coordinate grid
        filled = voxel_grid > threshold
        colors = plt.cm.get_cmap(colormap)(voxel_grid)
        
        ax.voxels(filled, facecolors=colors, edgecolors='gray', alpha=0.8, linewidth=0.1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set viewing angle
        elev, azim = view_angle
        ax.view_init(elev=elev, azim=azim)
        
        # Make axes equal
        max_range = voxel_grid.shape[0]
        ax.set_xlim([0, max_range])
        ax.set_ylim([0, max_range])
        ax.set_zlim([0, max_range])
        
        return filled.sum()  # Return number of filled voxels
    
    def visualize_training_samples(self, trainer, epoch, num_samples=4, 
                                   descriptions=None, embedding_cache=None,
                                   context_encoder=None, tokenizer=None, 
                                   granularity=32, device='cuda',
                                   view_angles=[(30, 45), (30, 135), (60, 45)],
                                   colormap='viridis', show_slices=True, 
                                   show_progress=False):
        """
        Generate and visualize samples from the model.
        Uses DDIM or layer convergence based on trainer's use_ddim flag.
        
        Args:
            trainer: LayerXLayerDiffusionTrainer instance
            epoch: Current epoch number
            num_samples: Number of samples to generate
            descriptions: Optional list of text descriptions (uses defaults if None)
            embedding_cache: T5EmbeddingCache instance for cached embeddings (preferred)
            context_encoder: T5 encoder for text conditioning (fallback if cache miss)
            tokenizer: T5 tokenizer (fallback if cache miss)
            granularity: Voxel grid resolution
            device: Device to run on
            view_angles: List of (elevation, azimuth) tuples for different views
            colormap: Matplotlib colormap to use
            show_slices: Whether to show 2D slice views
            show_progress: Whether to show progress bars
        """
        if show_progress:
            print(f"\n{'='*60}")
            print(f"GENERATING VISUALIZATION FOR EPOCH {epoch}")
            print(f"{'='*60}")
        
        # Use provided descriptions or defaults
        if descriptions is None:
            descriptions = [
                "A simple cube",
                "A sphere",
                "A pyramid",
                "An abstract shape"
            ][:num_samples]
        
        # Encode descriptions to context embeddings
        with torch.no_grad():
            if show_progress:
                print("Encoding text descriptions...")
            
            # Try to use cached embeddings first
            if embedding_cache is not None:
                try:
                    # Attempt to get cached embeddings
                    context = embedding_cache.get_batch_embeddings(descriptions, device)
                    if show_progress:
                        print("✓ Using cached T5 embeddings")
                except KeyError as e:
                    # Cache miss - fall back to CPU T5 model
                    if show_progress:
                        print(f"⚠ Cache miss detected: {e}")
                        print("⚠ Falling back to CPU T5 model...")
                    
                    if context_encoder is not None and tokenizer is not None:
                        # Move model to CPU if not already there
                        context_encoder = context_encoder.cpu()
                        
                        text_inputs = tokenizer(
                            list(descriptions),
                            padding='max_length',
                            max_length=77,
                            truncation=True,
                            return_tensors='pt'
                        )
                        
                        # Compute on CPU
                        context = context_encoder(
                            input_ids=text_inputs.input_ids,
                            attention_mask=text_inputs.attention_mask
                        ).last_hidden_state
                        
                        # Move context to target device
                        context = context.to(device)
                        
                        if show_progress:
                            print("✓ Generated embeddings using CPU T5 model")
                        
                        # Optionally cache the new embeddings for future use
                        for desc, emb in zip(descriptions, context):
                            embedding_cache.cache[desc] = emb.cpu()
                        embedding_cache.save_cache()
                        if show_progress:
                            print("✓ New embeddings cached for future use")
                    else:
                        if show_progress:
                            print("✗ No T5 model provided for fallback, using random context")
                        # Last resort: random context
                        context = torch.randn(num_samples, 77, 768).to(device)
            
            elif context_encoder is not None and tokenizer is not None:
                # No cache provided, use model directly
                if show_progress:
                    print("⚠ No cache provided, using T5 model directly...")
                
                # Use CPU to save GPU memory
                context_encoder = context_encoder.cpu()
                
                text_inputs = tokenizer(
                    list(descriptions),
                    padding='max_length',
                    max_length=77,
                    truncation=True,
                    return_tensors='pt'
                )
                
                context = context_encoder(
                    input_ids=text_inputs.input_ids,
                    attention_mask=text_inputs.attention_mask
                ).last_hidden_state
                
                # Move to target device
                context = context.to(device)
            else:
                # Fallback: random context if neither cache nor encoder provided
                if show_progress:
                    print("⚠ No cache or T5 model provided, using random context")
                context = torch.randn(num_samples, 77, 768).to(device)
            
            # Generate samples using the unified sample() method
            if show_progress:
                sampling_method = "DDIM" if trainer.use_ddim else "Layer Convergence"
                print(f"Generating {num_samples} voxel samples ({granularity}³ resolution) using {sampling_method}...")
                if trainer.use_ddim:
                    samples = self._generate_with_progress_ddim(
                        trainer, context, num_samples, granularity, device
                    )
                else:
                    samples = self._generate_with_progress(
                        trainer, context, num_samples, granularity, device
                    )
            else:
                samples = trainer.sample(
                    context,
                    shape=(num_samples, 1, granularity, granularity),
                    device=device
                )
            
            samples = samples.clamp(0, 1).cpu().numpy()
        
        # Create visualization
        if show_progress:
            print("Creating visualization plots...")
        
        num_views = len(view_angles)
        
        if show_slices:
            # Show both 3D views and 2D slices
            fig = plt.figure(figsize=(5 * (num_views + 1), 4 * num_samples))
            
            iterator = tqdm(range(num_samples), desc="Plotting samples", disable=not show_progress)
            for i in iterator:
                voxel_grid = samples[i, 0]  # [H, W, D]
                
                # 3D views
                for j, (elev, azim) in enumerate(view_angles):
                    ax = fig.add_subplot(num_samples, num_views + 1, 
                                        i * (num_views + 1) + j + 1, 
                                        projection='3d')
                    
                    num_voxels = self.visualize_single_voxel(
                        voxel_grid, ax, view_angle=(elev, azim),
                        colormap=colormap
                    )
                    
                    if j == 0:
                        # Truncate description for display
                        desc_short = (descriptions[i][:30] + '...') if len(descriptions[i]) > 30 else descriptions[i]
                        ax.set_title(f'{desc_short}\n{num_voxels:,} voxels\nView: {elev}°, {azim}°', 
                                   fontsize=10)
                    else:
                        ax.set_title(f'View: {elev}°, {azim}°', fontsize=10)
                
                # 2D slice view
                ax_slice = fig.add_subplot(num_samples, num_views + 1, 
                                          i * (num_views + 1) + num_views + 1)
                mid_slice = voxel_grid[:, :, granularity // 2]
                ax_slice.imshow(mid_slice, cmap='gray', vmin=0, vmax=1)
                ax_slice.set_title(f'Middle Slice (z={granularity//2})', fontsize=10)
                ax_slice.axis('off')
        else:
            # Only 3D views
            fig = plt.figure(figsize=(5 * num_views, 4 * num_samples))
            
            iterator = tqdm(range(num_samples), desc="Plotting samples", disable=not show_progress)
            for i in iterator:
                voxel_grid = samples[i, 0]  # [H, W, D]
                
                for j, (elev, azim) in enumerate(view_angles):
                    ax = fig.add_subplot(num_samples, num_views, 
                                        i * num_views + j + 1, 
                                        projection='3d')
                    
                    num_voxels = self.visualize_single_voxel(
                        voxel_grid, ax, view_angle=(elev, azim),
                        colormap=colormap
                    )
                    
                    if j == 0:
                        desc_short = (descriptions[i][:30] + '...') if len(descriptions[i]) > 30 else descriptions[i]
                        ax.set_title(f'{desc_short}\n{num_voxels:,} voxels', fontsize=10)
        
        sampling_method = "DDIM" if trainer.use_ddim else "Layer Convergence"
        plt.suptitle(f'Epoch {epoch} - Generated Voxel Samples ({sampling_method})', 
                    fontsize=14, y=0.995)
        plt.tight_layout()
        
        # Save figure
        save_path = self.save_dir / f'epoch_{epoch:04d}_samples.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show_progress:
            print(f"✓ Visualization saved to: {save_path}")
            print(f"{'='*60}\n")
        
        # Display in notebook
        plt.show()
        plt.close()
        
        return samples
    
    def _generate_with_progress(self, trainer, context, num_samples, granularity, device):
        """
        Generate samples with layer convergence and a progress bar showing layer-by-layer progress.
        This is a wrapper around the trainer's generation method.
        """
        trainer.model.eval()
        model_dtype = next(trainer.model.parameters()).dtype
        
        with torch.no_grad():
            batch_size = num_samples
            prev_layer_features = None
            shape = (num_samples, 1, granularity, granularity)
            
            # Initialize voxel grid with correct dimensions: [B, C, H, W, D]
            voxel_grid = torch.zeros((batch_size, shape[1], shape[2], shape[3], trainer.model.granularity), device=device)

            # Progress bar for layers
            layer_pbar = tqdm(range(trainer.model.granularity), desc="Generating layers", unit="layer")
            
            for layer_idx in layer_pbar:
                # Create a tensor for the current layer index
                l = torch.full((batch_size,), layer_idx, device=device, dtype=model_dtype)

                # Start with pure noise for this layer
                x = torch.randn(shape, device=device, dtype=model_dtype)

                # Progress bar for diffusion timesteps (nested)
                timestep_pbar = tqdm(
                    reversed(range(trainer.diffusion.timesteps)), 
                    desc=f"  Layer {layer_idx+1}/{trainer.model.granularity} diffusion",
                    leave=False,
                    total=trainer.diffusion.timesteps,
                    unit="step"
                )

                # Run diffusion to convergence for this layer
                for t in timestep_pbar:
                    t_batch = torch.full((batch_size,), t, device=device, dtype=model_dtype)
                    predicted_noise, prev_layer_features = trainer.model(x, t_batch, l, context, prev_layer_features)

                    alpha_t = trainer.diffusion.alphas[t]
                    alpha_hat_t = trainer.diffusion.alpha_hats[t]
                    beta_t = trainer.diffusion.betas[t]

                    if t > 0:
                        noise = torch.randn_like(x)
                    else:
                        noise = torch.zeros_like(x)

                    x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * predicted_noise) + torch.sqrt(beta_t) * noise
                
                timestep_pbar.close()

                # Place the generated layer into the voxel grid at the correct position
                voxel_grid[:, :, :, :, layer_idx] = x
                
                # Update layer progress bar with completion percentage
                layer_pbar.set_postfix({'completed': f'{((layer_idx+1)/trainer.model.granularity)*100:.1f}%'})

            layer_pbar.close()
            return voxel_grid.clamp(0, 1)
    
    def _generate_with_progress_ddim(self, trainer, context, num_samples, granularity, device):
        """
        Generate samples with DDIM and a progress bar showing layer-by-layer progress.
        """
        trainer.model.eval()
        model_dtype = next(trainer.model.parameters()).dtype
        with torch.no_grad():
            batch_size = num_samples
            prev_layer_features = None
            shape = (num_samples, 1, granularity, granularity)

            voxel_grid = torch.zeros((batch_size, shape[1], shape[2], shape[3],
                                      trainer.model.granularity), device=device, dtype=model_dtype)

            # Create subset of timesteps for DDIM
            timestep_interval = trainer.diffusion.timesteps // trainer.ddim_steps
            ddim_timesteps = list(range(0, trainer.diffusion.timesteps, timestep_interval))
            ddim_timesteps.reverse()
            
            # Progress bar for layers
            layer_pbar = tqdm(range(trainer.model.granularity), desc="Generating layers (DDIM)", unit="layer")
            
            for layer_idx in layer_pbar:
                l = torch.full((batch_size,), layer_idx, device=device, dtype=model_dtype)
                x = torch.randn(shape, device=device, dtype=model_dtype)

                # Progress bar for DDIM timesteps
                timestep_pbar = tqdm(
                    enumerate(ddim_timesteps),
                    desc=f"  Layer {layer_idx+1}/{trainer.model.granularity} DDIM",
                    leave=False,
                    total=len(ddim_timesteps),
                    unit="step"
                )
                
                for i, t in timestep_pbar:
                    t_batch = torch.full((batch_size,), t, device=device, dtype=model_dtype)
                    
                    predicted_noise, prev_layer_features = trainer.model(
                        x, t_batch, l, context, prev_layer_features
                    )
                    
                    # DDIM update rule
                    alpha_hat_t = trainer.diffusion.alpha_hats[t]
                    
                    # Predict x0
                    pred_x0 = (x - torch.sqrt(1 - alpha_hat_t) * predicted_noise) / torch.sqrt(alpha_hat_t)
                    pred_x0 = pred_x0.clamp(-1, 1)
                    
                    if i < len(ddim_timesteps) - 1:
                        t_prev = ddim_timesteps[i + 1]
                        alpha_hat_t_prev = trainer.diffusion.alpha_hats[t_prev]
                        
                        # Direction pointing to x_t
                        dir_xt = torch.sqrt(1 - alpha_hat_t_prev - trainer.ddim_eta**2) * predicted_noise
                        
                        # Add noise if eta > 0
                        if trainer.ddim_eta > 0:
                            noise = torch.randn_like(x) * trainer.ddim_eta
                        else:
                            noise = 0
                        
                        x = torch.sqrt(alpha_hat_t_prev) * pred_x0 + dir_xt + noise
                    else:
                        x = pred_x0
                
                timestep_pbar.close()
                voxel_grid[:, :, :, :, layer_idx] = x
                
                # Update layer progress bar with completion percentage
                layer_pbar.set_postfix({'completed': f'{((layer_idx+1)/trainer.model.granularity)*100:.1f}%'})
            
            layer_pbar.close()
            return voxel_grid.clamp(0, 1)
        

# Integration function to add to training loop
def visualize_epoch_samples(trainer, epoch, visualizer: TrainingVoxelVisualizer, dataset, 
                           embedding_cache=None, context_encoder=None, tokenizer=None, 
                           granularity=32, device='cuda', num_samples=4, show_progress=True):
    """
    Convenience function to call in training loop.
    Now includes support for T5 embedding cache with CPU fallback.
    
    Args:
        trainer: LayerXLayerDiffusionTrainer instance
        epoch: Current epoch number
        visualizer: TrainingVoxelVisualizer instance
        dataset: Dataset object with descriptions
        embedding_cache: T5EmbeddingCache instance (preferred for speed)
        context_encoder: Model for encoding text (e.g., T5) - used as fallback on CPU
        tokenizer: Tokenizer for text encoding - used as fallback
        granularity: Voxel grid resolution
        device: Device to run on
        num_samples: Number of samples to generate
        show_progress: Whether to show progress bars
    
    Example usage in training loop:
        if (epoch + 1) % 10 == 0:
            visualize_epoch_samples(
                trainer=layer_x_layer_trainer,
                epoch=epoch,
                visualizer=voxel_visualizer,
                dataset=dataloader_3d_val.dataset,
                embedding_cache=embedding_cache,  # Primary method
                context_encoder=t5_model,          # Fallback (on CPU)
                tokenizer=t5_tokenizer,            # Fallback
                granularity=granularity,
                device=device,
                num_samples=4,
                show_progress=True
            )
    """
    # Get random descriptions from dataset
    import random
    
    # Default descriptions if dataset has none
    default_descriptions = [
        "A 3D geometric shape",
        "An abstract voxel object",
        "A simple 3D structure",
        "A complex voxel model",
        "A creative 3D design",
        "An artistic voxel sculpture",
        "A minimalist 3D form",
        "A detailed voxel creation"
    ]
    
    # Try to get descriptions from dataset
    if hasattr(dataset, 'descriptions') and dataset.descriptions:
        available_descriptions = list(dataset.descriptions.values())
        
        if len(available_descriptions) >= num_samples:
            # Enough descriptions available, sample without replacement
            random_descriptions = random.sample(available_descriptions, k=num_samples)
        elif len(available_descriptions) > 0:
            # Some descriptions available, but not enough - use what we have and fill with defaults
            random_descriptions = available_descriptions[:num_samples]
            if len(random_descriptions) < num_samples:
                # Fill remaining with defaults
                num_needed = num_samples - len(random_descriptions)
                random_descriptions.extend(random.sample(default_descriptions, k=num_needed))
        else:
            # No descriptions available
            random_descriptions = random.sample(default_descriptions, k=num_samples)
    else:
        # Dataset doesn't have descriptions attribute
        random_descriptions = random.sample(default_descriptions, k=num_samples)
    
    visualizer.visualize_training_samples(
        trainer=trainer,
        epoch=epoch,
        num_samples=num_samples,
        descriptions=random_descriptions,
        embedding_cache=embedding_cache,  # Pass the cache
        context_encoder=context_encoder,  # Pass T5 model for fallback
        tokenizer=tokenizer,              # Pass tokenizer for fallback
        granularity=granularity,
        device=device,
        view_angles=[(30, 45), (60, 120)],  # Two different viewing angles
        colormap='viridis',
        show_slices=True,
        show_progress=show_progress
    )        

# %%

def create_distributed_dataloaders(train_dataset, val_dataset, batch_size):
    """
    Create DataLoaders with distributed sampling if in distributed mode.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size per GPU
        
    Returns:
        train_loader, val_loader, train_sampler
    """
    if is_distributed:
        # Create distributed samplers
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=42  # For reproducibility
        )
        
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False  # Don't shuffle validation
        )
        
        shuffle_train = False  # Sampler handles shuffling
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=shuffle_train if train_sampler is None else False,
        num_workers=4 * (world_size if is_distributed else 1),  # Scale workers with GPUs
        pin_memory=True,
        drop_last=True,  # Important for distributed training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=4 * (world_size if is_distributed else 1),
        pin_memory=True,
        drop_last=False,
    )
    
    return train_loader, val_loader, train_sampler


# %%
# train the layer x layer diffusion model

layer_context_dim = 64  # dimension of the layer context input

print_main("\n" + "="*60)
print_main("Initializing Models")
print_main("="*60)

layer_by_layer_convergence = True  # Whether to run diffusion to convergence for each layer
layer_diffusion = ForwardDiffusion(timesteps=1000, schedule='cosine')

print_main("Creating LayerXLayerDiffusionModelV2...")
layer_x_layer_model = LayerXLayerDiffusionModelV2(
    layer_context_dim=layer_context_dim,
    granularity=granularity,
    text_context_dim=768,      
    max_context_layers=16,    # Sliding window of 16 previous layers for context
    in_channels=1, 
    model_channels=512, 
    context_dim=768, 
    attention_resolutions=[8, 16, 32]  # Apply attention at 8x8, 16x16 and 32x32 feature maps
)

print_main("Initializing model weights...")
layer_x_layer_model = initialize_model_weights(layer_x_layer_model)
layer_x_layer_model = layer_x_layer_model.to(device)
print_main("✓ Model weights initialized")

# NOW wrap with FSDP (if enabled)
if args.shard_model:
    print_main("Wrapping model with FSDP...")
    fsdp_config = get_fsdp_config()
    layer_x_layer_model = FSDP(layer_x_layer_model, **fsdp_config)
    print_main("✓ Model wrapped with FSDP")
else:    
    print_main("Running without model sharding (single GPU)")

print_main("="*60 + "\n")

scheduler = None

layer_x_layer_trainer = LayerXLayerDiffusionTrainerV2(
    model=layer_x_layer_model,
    diffusion=layer_diffusion,
    scheduler=scheduler,
    layer_by_layer_convergence=layer_by_layer_convergence,
    teacher_forcing=True,       # NEW: Use ground truth layers
    use_ddim=True,              # NEW: Use DDIM for faster validation
    ddim_steps=50,              # NEW: Number of DDIM steps
    ddim_eta=0.0                # NEW: DDIM stochasticity (0.0 = deterministic)
)

# %%
batch_size = 4

# %%
train_dataset_3d = VoxelDataset(
    npy_folder_path=f'/home/ad.msoe.edu/benzshawelt/Kedziora/kedziora_research/description_data_generation/objaverse_parallel/objaverse_voxels/{granularity}', 
    description_folder_path='/home/ad.msoe.edu/benzshawelt/Kedziora/kedziora_research/description_data_generation/objaverse_parallel/objaverse_descriptions', 
    granularity=granularity, test_count=0  # Use full training set (set greater than 0 to limit)
)

val_dataset_3d = VoxelDataset(
    npy_folder_path=f'/home/ad.msoe.edu/benzshawelt/Kedziora/kedziora_research/description_data_generation/objaverse_parallel/objaverse_voxels/{granularity}', 
    description_folder_path='/home/ad.msoe.edu/benzshawelt/Kedziora/kedziora_research/description_data_generation/objaverse_parallel/objaverse_descriptions', 
    granularity=granularity, test_count=0  # Use 100 samples for validation (set greater than 0 to limit)
)

train_dataset_3d.save_miss_report("debug_misses.txt")

# Replace your existing dataloader creation with:
batch_size = args.batch_size if args.batch_size is not None else 4  # Your default batch size

dataloader_3d_train, dataloader_3d_val, train_sampler = create_distributed_dataloaders(
    train_dataset_3d,
    val_dataset_3d,
    batch_size
)

print_main(f"Batch size per GPU: {batch_size}")
print_main(f"Effective batch size: {batch_size * world_size}")


print(f"Training dataset size: {len(dataloader_3d_train.dataset)}")
print(f"Validation dataset size: {len(dataloader_3d_val.dataset)}")

# %%
from transformers import T5EncoderModel, T5Tokenizer
from tqdm import tqdm

# Initialize T5
t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
t5_model = T5EncoderModel.from_pretrained('t5-base').to(device)
t5_model.eval()
print(f"T5 model loaded with {sum(p.numel() for p in t5_model.parameters()):,} parameters")

# ========== SETUP T5 EMBEDDING CACHE ==========
print("\n" + "="*60)
print("Setting up T5 Embedding Cache")
print("="*60)

if rank == 0:
    embedding_cache = T5EmbeddingCacheSQL(cache_dir='./t5_cache', max_length=77, memory_cache_size=1_000)
    embedding_cache.load_cache()  # Load existing cache if available



    description_dataloader = DataLoader(
        train_dataset_3d, # Use training dataset descriptions for caching
        batch_size=128,  # Larger batch since we're discarding voxel data
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # Precompute embeddings using the DataLoader
    embedding_cache.precompute_embeddings(
        dataloader_or_texts=description_dataloader,
        tokenizer=t5_tokenizer,
        model=t5_model,
        device=device,
        batch_size=32  # Batch size for T5 embedding computation
    )

    embedding_cache.save_cache()
    del description_dataloader
    t5_model = t5_model.cpu()

    print("✓ T5 embeddings cached successfully.")

if is_distributed:
    dist.barrier()

if rank != 0:
    embedding_cache = T5EmbeddingCacheSQL(cache_dir='./t5_cache', max_length=77, memory_cache_size=1_000)
    embedding_cache.load_cache()

print("✓ T5 embeddings cached! Training will use cached embeddings for speed.\n")
torch.cuda.empty_cache()

voxel_visualizer = TrainingVoxelVisualizer(
    save_dir='./training_samples',
    max_voxels=30000  
)

# %%
n_epochs_3d = 500
visualization_interval = 25
validation_interval = 10  # Run validation every N epochs
use_ema_3d = False

# Track best model
best_val_loss = float('inf')
metrics_history = {
    'train_loss': [],
    'val_loss': [],
    'val_occupancy': [],
    'val_density': []
}

if use_ema_3d:
    ema_model_3d = ExponentialMovingAverage(layer_x_layer_model, decay=0.999)


n_epochs_3d = args.num_epochs
validation_interval = args.validation_interval

print_main("\n" + "="*60)
print_main("Starting Distributed Training")
print_main("="*60)
print_main(f"Total Epochs: {n_epochs_3d}")
print_main(f"Validation Interval: {validation_interval}")
print_main("="*60 + "\n")

torch.cuda.empty_cache()

for epoch in range(n_epochs_3d):
    if is_distributed and train_sampler is not None:
        train_sampler.set_epoch(epoch)
    
    # ========== TRAINING ==========
    epoch_loss = 0
    num_batches = 0
    
    # Memory tracking (only on rank 0)
    if rank == 0:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    layer_x_layer_trainer.model.train()
    
    # Use tqdm only on main process to avoid cluttered output
    if rank == 0:
        pbar = tqdm(dataloader_3d_train, desc=f"Epoch {epoch+1}/{n_epochs_3d}")
    else:
        pbar = dataloader_3d_train
    
    for x0, description in pbar:
        if is_distributed:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float16
        x0 = x0.to(device, dtype=dtype, non_blocking=True)
        
        # Get cached context embeddings
        context = embedding_cache.get_batch_embeddings(description, device)
        
        # Training step
        loss, layer_losses = layer_x_layer_trainer.train_step(x0, context)
        epoch_loss += loss
        num_batches += 1
        
        # Update progress bar (only on rank 0)
        if rank == 0:
            pbar.set_postfix({'loss': f'{loss:.4f}'})
        
        if use_ema_3d:
            ema_model_3d.update()
    
    # ========== SYNCHRONIZE METRICS ACROSS ALL GPUS ==========
    if is_distributed:
        # Convert to tensors for all_reduce
        metrics_tensor = torch.tensor([epoch_loss, float(num_batches)], device=device)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        
        epoch_loss = metrics_tensor[0].item()
        num_batches = int(metrics_tensor[1].item())
    
    avg_train_loss = epoch_loss / num_batches
    metrics_history['train_loss'].append(avg_train_loss)
    
    # Print epoch summary (only from main process)
    if rank == 0:
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"\nEpoch {epoch+1}/{n_epochs_3d} Summary:")
        print(f"  Train Loss:  {avg_train_loss:.4f}")
        print(f"  Peak Memory: {peak_memory:.2f} GB")
    
    # ========== VALIDATION (only on main process) ==========
    if (epoch + 1) % validation_interval == 0:
        if rank == 0:
            print(f"\nRunning validation...")
            torch.cuda.empty_cache()
            
            if use_ema_3d:
                ema_model_3d.apply_shadow()
            
            # Run validation
            val_metrics = layer_x_layer_trainer.validate(
                dataloader=dataloader_3d_val,
                context_cache=embedding_cache,
                device=device,
                max_batches=25
            )
            
            # Print and track metrics
            print(f"Validation Results:")
            print(f"  Val Loss:      {val_metrics['val_loss']:.4f}")
            print(f"  Val Occupancy: {val_metrics['val_occupancy']:.2%}")
            print(f"  Val Density:   {val_metrics['val_density']:.4f}")
            
            metrics_history['val_loss'].append(val_metrics['val_loss'])
            metrics_history['val_occupancy'].append(val_metrics['val_occupancy'])
            metrics_history['val_density'].append(val_metrics['val_density'])
            
            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                
                # Save model with proper FSDP handling
                if args.shard_model:
                    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                    with FSDP.state_dict_type(
                        layer_x_layer_model,
                        StateDictType.FULL_STATE_DICT,
                        save_policy
                    ):
                        model_state = layer_x_layer_model.state_dict()
                else:
                    model_state = layer_x_layer_model.state_dict()
                
                torch.save(model_state, 'best_model.pth')
                print(f"  ✓ New best model saved! (loss: {best_val_loss:.4f})")
            
            if use_ema_3d:
                ema_model_3d.restore()
        
        # Synchronize all processes after validation
        if is_distributed:
            dist.barrier()
    
    # ========== VISUALIZATION (only on main process) ==========
    if (epoch + 1) % visualization_interval == 0 and rank == 0:
        if use_ema_3d:
            ema_model_3d.apply_shadow()
        
        visualize_epoch_samples(
            trainer=layer_x_layer_trainer,
            epoch=epoch + 1,
            visualizer=voxel_visualizer,
            dataset=dataloader_3d_val.dataset,
            context_encoder=t5_model,
            tokenizer=t5_tokenizer,
            granularity=granularity,
            device=device,
            num_samples=4,
            show_progress=True
        )
        
        if use_ema_3d:
            ema_model_3d.restore()
    
    # ========== CHECKPOINT SAVING (only on main process) ==========
    if (epoch + 1) % args.checkpoint_interval == 0 and rank == 0:
        # Save model state with proper FSDP handling
        if args.shard_model:
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                layer_x_layer_model,
                StateDictType.FULL_STATE_DICT,
                save_policy
            ):
                model_state = layer_x_layer_model.state_dict()
        else:
            model_state = layer_x_layer_model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': layer_x_layer_trainer.optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'best_val_loss': best_val_loss,
            'metrics_history': metrics_history,
            'args': vars(args),  # Save training arguments
        }
        
        torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pth')
        print(f"  ✓ Checkpoint saved at epoch {epoch+1}")
    
    # Synchronize all processes before next epoch
    if is_distributed:
        dist.barrier()

if is_distributed:
    print_main("\nCleaning up distributed training...")
    dist.destroy_process_group()
    print_main("✓ Distributed training cleaned up")

print_main("\n" + "="*60)
print_main("Training Complete!")
print_main("="*60)

# %%
# Save final model and metrics
print("\n" + "="*60)
print("Training Complete!")
print("="*60)

torch.save(layer_x_layer_model.state_dict(), 'layer_x_layer_diffusion_model_final.pth')
print("✓ Final model saved to: layer_x_layer_diffusion_model_final.pth")

# Save metrics history
with open('training_metrics.json', 'w') as f:
    json.dump(metrics_history, f, indent=2)
print("✓ Training metrics saved to: training_metrics.json")

print(f"✓ Best validation loss: {best_val_loss:.4f}")
print(f"✓ Best model saved at: best_model.pth")

# %% 
def load_checkpoint_fsdp(model, checkpoint_path, is_fsdp=False):
    """
    Load checkpoint with proper FSDP handling.
    
    Args:
        model: Your model (wrapped with FSDP or not)
        checkpoint_path: Path to checkpoint file
        is_fsdp: Whether model is currently wrapped with FSDP
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if is_fsdp:
        # For FSDP models, use special loading
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=False)
        ):
            model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Regular loading for non-FSDP models
        model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Checkpoint loaded from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Train Loss: {checkpoint.get('train_loss', 'unknown'):.4f}")
    
    return checkpoint
# %%
# Plot training curves
print("\nGenerating training curves...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Loss curve
axes[0].plot(metrics_history['train_loss'], label='Train Loss', alpha=0.7)
if metrics_history['val_loss']:
    val_epochs = list(range(validation_interval, len(metrics_history['train_loss']), validation_interval))
    axes[0].plot(val_epochs, metrics_history['val_loss'], label='Val Loss', marker='o')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Occupancy curve
if metrics_history['val_occupancy']:
    axes[1].plot(val_epochs, metrics_history['val_occupancy'], marker='o', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Occupancy Rate')
    axes[1].set_title('Validation Voxel Occupancy')
    axes[1].grid(True, alpha=0.3)

# Density curve
if metrics_history['val_density']:
    axes[2].plot(val_epochs, metrics_history['val_density'], marker='o', color='orange')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Mean Density')
    axes[2].set_title('Validation Voxel Density')
    axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
print("✓ Training curves saved to: training_curves.png")
plt.show()

print("\n" + "="*60)
print("All Done! 🎉")
print("="*60)
print("\nKey files saved:")
print("  - best_model.pth (best model based on validation)")
print("  - layer_x_layer_diffusion_model_final.pth (final model)")
print("  - training_metrics.json (complete metrics history)")
print("  - training_curves.png (training visualization)")
print("  - ./training_samples/ (generated sample visualizations)")
print("  - ./t5_cache/ (cached T5 embeddings for future use)")
print("\n" + "="*60)