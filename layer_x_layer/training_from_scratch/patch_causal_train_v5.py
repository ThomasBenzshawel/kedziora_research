# %%
"""
Layer-by-Layer 3D Diffusion Model with Hybrid Causal Attention (V5)
===================================================================
This script integrates the Hybrid Causal Attention mechanism which combines:
- Factorized Attention (efficient, full resolution)  
- Patch-Based Attention (precise, downsampled for efficiency)
- Learned gating to balance both pathways

Key improvements over V4:
- Hybrid attention combining factorized + patch-based approaches
- Richer input encoding
- Gate monitoring for training analysis
"""

from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union
import math
import random
import os
import json
from difflib import get_close_matches
from functools import lru_cache

import numpy as np
from numpy.random import RandomState

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data.distributed import DistributedSampler

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import argparse
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

from functools import partial
import h5py
from transformers import T5EncoderModel, T5Tokenizer
from tqdm import tqdm
import hashlib
import sqlite3


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    """Parse command line arguments for training configuration"""
    parser = argparse.ArgumentParser(description='Train 3D Layer-by-Layer Diffusion Model V5')
    
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
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Override batch size (per GPU)')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--validation_interval', type=int, default=10,
                        help='Run validation every N epochs')
    parser.add_argument('--checkpoint_interval', type=int, default=50,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--current_layer_only', action='store_true',
                        help='Use only current layer low-res context (vs all layers)')
    parser.add_argument('--enable_fuzzy_matching', action='store_true',
                        help='Enable fuzzy matching for missing descriptions (slower initialization)')
    parser.add_argument('--skip_cache_scan', action='store_true',
                        help='Skip scanning datasets for uncached T5 embeddings (use when cache is complete)')
    parser.add_argument('--granularities', type=int, nargs='+', default=[16, 32, 64],
                        help='List of granularities for progressive training (e.g., 16 32 64)')
    parser.add_argument('--epochs_per_stage', type=int, nargs='+', default=[500, 250, 200],
                        help='Number of epochs for each granularity stage (e.g., 500 250 200)')
    
    # V5 Hybrid Attention parameters
    parser.add_argument('--hybrid_num_heads', type=int, default=8,
                        help='Number of attention heads in hybrid attention')
    parser.add_argument('--hybrid_patch_size', type=int, default=4,
                        help='Patch size for patch-based attention pathway')
    parser.add_argument('--hybrid_dropout', type=float, default=0.1,
                        help='Dropout rate for hybrid attention')

    return parser.parse_args()


# Parse arguments
args = parse_args()

granularity = 16  # Initial granularity


# ============================================================================
# DISTRIBUTED SETUP
# ============================================================================

def setup_distributed():
    """
    Initialize distributed training environment for FSDP.
    Returns: (is_distributed, rank, world_size, local_rank)
    """
    if not args.shard_model:
        return False, 0, 1, 0
    
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        print("ERROR: --shard_model requires running with torchrun or torch.distributed.launch")
        print("Example: torchrun --nproc_per_node=auto train.py --shard_model")
        exit(1)
    
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    torch.cuda.set_device(local_rank)
    
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
    )
    
    return True, rank, world_size, local_rank


is_distributed, rank, world_size, local_rank = setup_distributed()

if is_distributed:
    device = torch.device(f"cuda:{local_rank}")
    print(f"[Rank {rank}/{world_size}] Running on GPU {local_rank}")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on single device: {device}")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')


def print_main(*args_print, **kwargs):
    """Print only from rank 0 (main process)"""
    if not is_distributed or rank == 0:
        print(*args_print, **kwargs)


def get_fsdp_config():
    """Get FSDP configuration based on command-line arguments."""
    strategy_map = {
        'FULL_SHARD': ShardingStrategy.FULL_SHARD,
        'SHARD_GRAD_OP': ShardingStrategy.SHARD_GRAD_OP,
        'NO_SHARD': ShardingStrategy.NO_SHARD,
    }
    sharding_strategy = strategy_map[args.sharding_strategy]
    
    cpu_offload = CPUOffload(offload_params=True) if args.cpu_offload else None
    
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
        min_num_params=1_000_000
    )
    
    config = {
        'sharding_strategy': sharding_strategy,
        'cpu_offload': cpu_offload,
        'mixed_precision': mixed_precision,
        'auto_wrap_policy': auto_wrap_policy,
        'backward_prefetch': BackwardPrefetch.BACKWARD_PRE,
        'device_id': torch.cuda.current_device(),
        'sync_module_states': True,
    }
    
    print_main("FSDP Configuration:")
    print_main(f"  Sharding Strategy: {args.sharding_strategy}")
    print_main(f"  CPU Offload: {args.cpu_offload}")
    print_main(f"  Mixed Precision: {args.mixed_precision}")
    print_main(f"  World Size: {world_size} GPUs")
    
    return config


MASTER_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def initialize_model_weights(model):
    """Initialize model weights properly before FSDP wrapping."""
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm2d)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=0.02)
    
    model.apply(init_weights)
    return model


# ============================================================================
# T5 EMBEDDING CACHE
# ============================================================================

class T5EmbeddingCache:
    """Unified T5 embedding cache using SQLite backend with LRU memory cache."""
    
    def __init__(self, cache_dir='./t5_cache', max_length=77, memory_cache_size=1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_length = max_length
        self.db_path = self.cache_dir / 'embeddings.db'
        self.memory_cache_size = memory_cache_size
        self.tokenizer = None
        self.model = None
        
        self._memory_cache = {}
        self._cache_order = []
        
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database with optimizations"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                text_hash TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_text_hash 
            ON embeddings(text_hash)
        ''')
        
        cursor.execute('PRAGMA journal_mode=WAL')
        cursor.execute('PRAGMA synchronous=NORMAL')
        cursor.execute('PRAGMA cache_size=-64000')
        
        conn.commit()
        conn.close()
    
    def _get_cache_key(self, text):
        return hashlib.md5(text.encode()).hexdigest()
    
    def _add_to_memory_cache(self, key, embedding):
        if key in self._memory_cache:
            self._cache_order.remove(key)
        
        self._memory_cache[key] = embedding
        self._cache_order.append(key)
        
        if len(self._memory_cache) > self.memory_cache_size:
            oldest_key = self._cache_order.pop(0)
            del self._memory_cache[oldest_key]
    
    def load_cache(self):
        if self.db_path.exists():
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM embeddings')
            count = cursor.fetchone()[0]
            conn.close()
            
            if count > 0:
                print(f"SQLite cache ready with {count:,} cached embeddings")
            else:
                print("No existing cache found, will create new cache")
        else:
            print("No existing cache found, will create new cache")
    
    def save_cache(self):
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM embeddings')
        count = cursor.fetchone()[0]
        conn.close()
        print(f"Database contains {count:,} cached embeddings")
    
    def get_embedding(self, text):
        key = self._get_cache_key(text)
        
        if key in self._memory_cache:
            return self._memory_cache[key]
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute('SELECT embedding FROM embeddings WHERE text_hash = ?', (key,))
        result = cursor.fetchone()
        conn.close()
        
        if result is not None:
            embedding = torch.from_numpy(
                np.frombuffer(result[0], dtype=np.float16).reshape(self.max_length, 768).copy()
            )
            self._add_to_memory_cache(key, embedding)
            return embedding
        
        print(f"Cache miss for text: {text[:50]}...")
        if self.model is None or self.tokenizer is None:
            return None
        else:
            model_device = next(self.model.parameters()).device
            
            text_input = self.tokenizer(
                text,
                padding='max_length',
                max_length=self.max_length,
                truncation=True,
                return_tensors='pt'
            ).to(model_device)
            
            embedding = self.model(
                input_ids=text_input.input_ids,
                attention_mask=text_input.attention_mask
            ).last_hidden_state
            
            self.add_embedding(text, embedding.squeeze(0))
            del text_input
            embedding_cpu = embedding.squeeze(0).cpu()
            return embedding_cpu

    def add_embedding(self, text, embedding):
        key = self._get_cache_key(text)
        embedding_bytes = embedding.cpu().detach().numpy().astype(np.float16).tobytes()
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO embeddings (text_hash, embedding)
            VALUES (?, ?)
        ''', (key, embedding_bytes))
        conn.commit()
        conn.close()
        
        self._add_to_memory_cache(key, embedding.cpu())
    
    def precompute_embeddings_from_dataset(self, dataset, tokenizer, model, device, batch_size=32):
        print(f"Scanning {len(dataset)} samples for uncached descriptions...")
        
        uncached_descriptions = set()
        
        for idx in tqdm(range(len(dataset)), desc="Scanning for uncached descriptions"):
            _, description, file_idx = dataset[idx]
            if self.get_embedding(description) is None:
                uncached_descriptions.add(description)
        
        if not uncached_descriptions:
            print("All embeddings already cached!")
            return
        
        print(f"Found {len(uncached_descriptions)} unique uncached descriptions")
        print(f"Computing embeddings...")
        
        uncached_list = list(uncached_descriptions)
        model.eval()
        
        with torch.no_grad():
            for i in tqdm(range(0, len(uncached_list), batch_size), desc="Caching embeddings"):
                batch_texts = uncached_list[i:i+batch_size]
                
                text_inputs = tokenizer(
                    batch_texts,
                    padding='max_length',
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors='pt'
                ).to(device)
                
                embeddings = model(
                    input_ids=text_inputs.input_ids,
                    attention_mask=text_inputs.attention_mask
                ).last_hidden_state
                
                for text, emb in zip(batch_texts, embeddings):
                    self.add_embedding(text, emb)
                
                del text_inputs, embeddings
                if torch.cuda.is_available() and (i // batch_size) % 10 == 0:
                    torch.cuda.empty_cache()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"✓ Cached {len(uncached_list)} embeddings to {self.db_path}")
    
    def precompute_embeddings(self, texts, tokenizer, model, device, batch_size=128):
        if hasattr(texts, '__getitem__') and hasattr(texts, '__len__') and not isinstance(texts, (list, tuple, set)):
            return self.precompute_embeddings_from_dataset(texts, tokenizer, model, device, batch_size)
        
        uncached_texts = [t for t in texts if self.get_embedding(t) is None]
        
        if not uncached_texts:
            print("All embeddings already cached!")
            return
        
        print(f"Computing embeddings for {len(uncached_texts)} uncached texts...")
        model.eval()
        
        with torch.no_grad():
            for i in tqdm(range(0, len(uncached_texts), batch_size), desc="Caching embeddings"):
                batch_texts = uncached_texts[i:i+batch_size]
                
                text_inputs = tokenizer(
                    batch_texts,
                    padding='max_length',
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors='pt'
                ).to(device)
                
                embeddings = model(
                    input_ids=text_inputs.input_ids,
                    attention_mask=text_inputs.attention_mask
                ).last_hidden_state
                
                for text, emb in zip(batch_texts, embeddings):
                    self.add_embedding(text, emb)
                
                del text_inputs, embeddings
                if torch.cuda.is_available() and (i // batch_size) % 10 == 0:
                    torch.cuda.empty_cache()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"✓ Cached {len(uncached_texts)} embeddings to {self.db_path}")
    
    def get_batch_embeddings(self, texts, device):
        embeddings = []
        for text in texts:
            emb = self.get_embedding(text)
            if emb is None:
                raise ValueError(f"Embedding not found in cache for text: {text[:50]}...")
            embeddings.append(emb)
        return torch.stack(embeddings).to(device, dtype=MASTER_DTYPE)

    def set_embedding_model(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer


def load_pretrained_encoder(model, encoder_checkpoint_path):
    """Load pretrained encoder weights into the model"""
    checkpoint = torch.load(encoder_checkpoint_path, map_location='cpu')
    encoder_state_dict = checkpoint['encoder_state_dict']
    
    if any(k.startswith('_orig_mod.') for k in encoder_state_dict.keys()):
        encoder_state_dict = {
            k.replace('_orig_mod.', ''): v 
            for k, v in encoder_state_dict.items()
        }
    
    model.low_res_encoder.load_state_dict(encoder_state_dict, strict=True)
    
    for param in model.low_res_encoder.parameters():
        param.requires_grad = False
    
    print(f"✓ Loaded pretrained encoder from {encoder_checkpoint_path}")
    print("  Encoder weights frozen for training")
    
    return model


# ============================================================================
# VALIDATION METRICS
# ============================================================================

class ValidationMetrics:
    """Track and compute validation metrics for 3D generation"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.losses = []
        self.layer_losses = []
        self.voxel_occupancy = []
        self.voxel_density = []
    
    def update(self, loss, layer_losses=None, generated_voxels=None):
        self.losses.append(loss)
        
        if layer_losses is not None:
            self.layer_losses.append(layer_losses)
        
        if generated_voxels is not None:
            batch_size = generated_voxels.shape[0]
            for b in range(batch_size):
                voxels = generated_voxels[b].cpu().numpy()
                occupancy = (voxels > 0.5).mean()
                density = voxels.mean()
                
                self.voxel_occupancy.append(occupancy)
                self.voxel_density.append(density)
    
    def compute(self):
        metrics = {
            'val_loss': np.mean(self.losses) if self.losses else float('inf'),
            'val_occupancy': np.mean(self.voxel_occupancy) if self.voxel_occupancy else 0.0,
            'val_density': np.mean(self.voxel_density) if self.voxel_density else 0.0,
        }
        
        if self.layer_losses:
            layer_losses_array = np.array(self.layer_losses)
            metrics['val_layer_losses'] = layer_losses_array.mean(axis=0).tolist()
        
        return metrics


# ============================================================================
# CORE ATTENTION MODULES
# ============================================================================

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

        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

        assert channels % num_heads == 0

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        
        h = self.norm(x)
        
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        k = k.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        v = v.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(2, 3).contiguous().view(B, C, H, W)
        
        out = self.proj(out)
        return x + out


class MultiHeadCrossAttention(nn.Module):
    """Cross-attention from spatial features to context sequence"""
    def __init__(self, channels, context_dim, num_heads=2):
        super().__init__()
        self.channels = channels
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.to_kv = nn.Linear(context_dim, channels * 2)
        self.proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x, context):
        B, C, H, W = x.shape
        
        h = self.norm(x)
        
        q = self.q(h)
        q = q.view(B, self.num_heads, self.head_dim, H * W)
        q = q.permute(0, 1, 3, 2)
        
        seq_len = context.shape[1]
        kv = self.to_kv(context)
        kv = kv.view(B, seq_len, 2, self.num_heads, self.head_dim)
        k, v = kv[:, :, 0], kv[:, :, 1]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).contiguous()
        out = out.view(B, C, H, W)
        
        out = self.proj(out)
        return x + out


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
        scale_shift_out = self.scale_shift(cond)
        scale, shift = scale_shift_out.chunk(2, dim=-1)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


class ResnetBlockWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, context_dim, 
                 layer_context_dim=64, use_attention=False, num_heads=8, num_groups=8):
        super().__init__()
        
        cond_dim = time_emb_dim + layer_context_dim
        
        self.norm1 = AdaGN(num_groups, in_channels, cond_dim)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.norm2 = AdaGN(num_groups, out_channels, cond_dim)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.activation = nn.SiLU()
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = TransformerBlock(out_channels, context_dim, num_heads=num_heads)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, t, context_emb, layer_context=None):
        if layer_context is not None:
            cond = torch.cat([t, layer_context], dim=-1)
        else:
            layer_zeros = torch.zeros(t.shape[0], self.norm1.scale_shift.in_features - t.shape[1], 
                                     device=t.device, dtype=t.dtype)
            cond = torch.cat([t, layer_zeros], dim=-1)
        
        h = self.norm1(x, cond)
        h = self.activation(h)
        h = self.conv1(h)
        
        h = self.norm2(h, cond)
        h = self.activation(h)
        h = self.conv2(h)
        
        if self.use_attention:
            h = self.attention(h, context_emb)
        
        return h + self.shortcut(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        model_dtype = x.dtype
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=model_dtype) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1, 0, 0))
        return emb


# ============================================================================
# HYBRID CAUSAL ATTENTION COMPONENTS (V5)
# ============================================================================

class PatchEmbedding(nn.Module):
    """Convert spatial features into patch tokens (like Vision Transformer)."""
    def __init__(self, channels, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.channels = channels
        
        self.proj = nn.Conv2d(
            channels, channels,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = nn.LayerNorm(channels)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        patches = self.proj(x)
        _, _, pH, pW = patches.shape
        
        patches = patches.flatten(2).transpose(1, 2)
        patches = self.norm(patches)
        
        return patches, (pH, pW)


class PatchUnembedding(nn.Module):
    """Convert patch tokens back to spatial features."""
    def __init__(self, channels, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.channels = channels
        
        self.proj = nn.ConvTranspose2d(
            channels, channels,
            kernel_size=patch_size,
            stride=patch_size
        )
        
    def forward(self, patches, patch_shape):
        B, N, C = patches.shape
        pH, pW = patch_shape
        
        x = patches.transpose(1, 2).reshape(B, C, pH, pW)
        x = self.proj(x)
        
        return x


class PatchBasedCausalAttention(nn.Module):
    """
    Full patch-to-patch attention across layers with causal masking.
    Each patch in layer i can attend to ALL patches in layers 0..i-1.
    """
    def __init__(self, channels, num_heads=8, patch_size=4, dropout=0.1, max_layers=512):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.patch_size = patch_size
        
        assert channels % num_heads == 0
        
        self.patch_embed = PatchEmbedding(channels, patch_size)
        self.patch_unembed = PatchUnembedding(channels, patch_size)
        
        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)
        
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.Linear(channels, channels)
        
        self.layer_pos_embed = nn.Embedding(max_layers, channels)
        self.patch_pos_embed = nn.Embedding(4096, channels)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x_noisy, x_clean):
        B, L, C, H, W = x_noisy.shape
        device = x_noisy.device
        
        x_noisy_flat = x_noisy.reshape(B * L, C, H, W)
        x_clean_flat = x_clean.reshape(B * L, C, H, W)
        
        patches_noisy, patch_shape = self.patch_embed(x_noisy_flat)
        patches_clean, _ = self.patch_embed(x_clean_flat)
        
        P = patches_noisy.shape[1]
        pH, pW = patch_shape
        
        patches_noisy = patches_noisy.reshape(B, L, P, C)
        patches_clean = patches_clean.reshape(B, L, P, C)
        
        layer_indices = torch.arange(L, device=device)
        patch_indices = torch.arange(P, device=device)
        
        layer_pos = self.layer_pos_embed(layer_indices).view(1, L, 1, C)
        patch_pos = self.patch_pos_embed(patch_indices).view(1, 1, P, C)
        
        patches_clean = patches_clean + layer_pos + patch_pos
        
        patches_noisy = self.norm_q(patches_noisy)
        patches_clean = self.norm_kv(patches_clean)
        
        q = self.to_q(patches_noisy)
        k = self.to_k(patches_clean)
        v = self.to_v(patches_clean)
        
        q = q.reshape(B, L * P, C)
        k = k.reshape(B, L * P, C)
        v = v.reshape(B, L * P, C)
        
        q = q.view(B, L * P, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L * P, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L * P, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        token_layer_ids = torch.arange(L, device=device).unsqueeze(1).expand(-1, P).flatten()
        causal_mask = token_layer_ids.unsqueeze(1) <= token_layer_ids.unsqueeze(0)
        
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().reshape(B, L * P, C)
        out = self.to_out(out)
        
        out = out.reshape(B, L, P, C)
        
        out_flat = out.reshape(B * L, P, C)
        out_spatial = self.patch_unembed(out_flat, patch_shape)
        out_spatial = out_spatial.reshape(B, L, C, H, W)
        
        return x_noisy + out_spatial


class CausalLayerAttention(nn.Module):
    """
    Layer-level causal attention: determines WHICH previous layers to attend to.
    Operates on pooled layer representations for efficiency.
    """
    def __init__(self, channels, num_heads=8, dropout=0.1, max_layers=512):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0
        
        self.norm_q = nn.LayerNorm(channels)
        self.norm_k = nn.LayerNorm(channels)
        
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        
        self.layer_pos_embed = nn.Embedding(max_layers, channels)
        self.relative_pos_bias = nn.Embedding(2 * max_layers, num_heads)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x_noisy, x_clean):
        B, L, C, H, W = x_noisy.shape
        device = x_noisy.device
        
        q_pooled = x_noisy.mean(dim=[-2, -1])
        k_pooled = x_clean.mean(dim=[-2, -1])
        
        layer_indices = torch.arange(L, device=device)
        layer_pos = self.layer_pos_embed(layer_indices)
        k_pooled = k_pooled + layer_pos.unsqueeze(0)
        
        q = self.to_q(self.norm_q(q_pooled))
        k = self.to_k(self.norm_k(k_pooled))
        
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        query_pos = layer_indices.unsqueeze(1)
        key_pos = layer_indices.unsqueeze(0)
        relative_distance = (query_pos - key_pos) + L
        relative_distance = relative_distance.clamp(0, 2 * L - 1)
        
        rel_bias = self.relative_pos_bias(relative_distance)
        rel_bias = rel_bias.permute(2, 0, 1).unsqueeze(0)
        attn = attn + rel_bias
        
        causal_mask = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=0)
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        
        return attn.mean(dim=1)


class SpatialCrossAttention(nn.Module):
    """
    Spatial cross-attention: determines WHERE to look in the aggregated context.
    """
    def __init__(self, channels, num_heads=8, dropout=0.1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0
        
        num_groups = min(8, channels)
        while channels % num_groups != 0:
            num_groups -= 1
        self.norm_q = nn.GroupNorm(num_groups, channels)
        self.norm_kv = nn.GroupNorm(num_groups, channels)
        
        self.to_q = nn.Conv2d(channels, channels, 1)
        self.to_k = nn.Conv2d(channels, channels, 1)
        self.to_v = nn.Conv2d(channels, channels, 1)
        self.to_out = nn.Conv2d(channels, channels, 1)
        
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, channels, 64, 64) * 0.02)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, query, context):
        B, C, H, W = query.shape
        
        if H != self.spatial_pos_embed.shape[2] or W != self.spatial_pos_embed.shape[3]:
            spatial_pos = F.interpolate(
                self.spatial_pos_embed, size=(H, W), 
                mode='bilinear', align_corners=False
            )
        else:
            spatial_pos = self.spatial_pos_embed
        
        q = self.to_q(self.norm_q(query))
        k = self.to_k(self.norm_kv(context + spatial_pos))
        v = self.to_v(self.norm_kv(context))
        
        q = q.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        k = k.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        v = v.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        out = self.to_out(out)
        
        return out


class FactorizedCausalAttention(nn.Module):
    """
    Factorized Causal Attention: Layer attention + Spatial attention.
    More efficient than patch-based: O(L² + H²W²) vs O(L²P²)
    """
    def __init__(self, channels, num_heads=8, dropout=0.1):
        super().__init__()
        self.channels = channels
        
        self.layer_attn = CausalLayerAttention(channels, num_heads, dropout)
        self.spatial_attn = SpatialCrossAttention(channels, num_heads, dropout)
        self.output_scale = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x_noisy, x_clean):
        B, L, C, H, W = x_noisy.shape
        
        layer_weights = self.layer_attn(x_noisy, x_clean)
        
        x_clean_flat = x_clean.view(B, L, -1)
        context = torch.bmm(layer_weights, x_clean_flat)
        context = context.view(B, L, C, H, W)
        
        x_noisy_flat = x_noisy.view(B * L, C, H, W)
        context_flat = context.view(B * L, C, H, W)
        
        spatial_out = self.spatial_attn(x_noisy_flat, context_flat)
        spatial_out = spatial_out.view(B, L, C, H, W)
        
        return x_noisy + self.output_scale * spatial_out


class HybridCausalAttention(nn.Module):
    """
    Hybrid Causal Attention: Combines Factorized + Patch-based attention.
    
    - Factorized pathway: Efficient, operates at full resolution
    - Patch pathway: Precise, operates at reduced resolution for efficiency
    
    The model learns to balance both pathways via a learned gate.
    """
    def __init__(self, channels, num_heads=8, patch_size=4, 
                 dropout=0.1, patch_downsample=2):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.patch_downsample = patch_downsample
        
        # Factorized Pathway (Full Resolution)
        self.factorized_attn = FactorizedCausalAttention(
            channels=channels,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Patch-Based Pathway (Downsampled)
        self.patch_downsample_layer = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=patch_downsample, padding=1),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU()
        ) if patch_downsample > 1 else nn.Identity()
        
        self.patch_attn = PatchBasedCausalAttention(
            channels=channels,
            num_heads=num_heads,
            patch_size=patch_size,
            dropout=dropout
        )
        
        self.patch_upsample_layer = nn.Sequential(
            nn.Upsample(scale_factor=patch_downsample, mode='bilinear', align_corners=False),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU()
        ) if patch_downsample > 1 else nn.Identity()
        
        # Gating Mechanism
        self.gate_network = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.SiLU(),
            nn.Linear(channels // 4, 2),
            nn.Softmax(dim=-1)
        )
        
        # Output Fusion
        self.output_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        
        self.output_scale = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x_noisy, x_clean):
        B, L, C, H, W = x_noisy.shape
        device = x_noisy.device
        
        # Pathway 1: Factorized Attention (Full Resolution)
        out_factorized = self.factorized_attn(x_noisy, x_clean)
        
        # Pathway 2: Patch-Based Attention (Downsampled)
        x_noisy_flat = x_noisy.reshape(B * L, C, H, W)
        x_clean_flat = x_clean.reshape(B * L, C, H, W)
        
        x_noisy_down = self.patch_downsample_layer(x_noisy_flat)
        x_clean_down = self.patch_downsample_layer(x_clean_flat)
        
        _, _, H_down, W_down = x_noisy_down.shape
        
        x_noisy_down = x_noisy_down.reshape(B, L, C, H_down, W_down)
        x_clean_down = x_clean_down.reshape(B, L, C, H_down, W_down)
        
        out_patch_down = self.patch_attn(x_noisy_down, x_clean_down)
        
        out_patch_flat = out_patch_down.reshape(B * L, C, H_down, W_down)
        out_patch = self.patch_upsample_layer(out_patch_flat)
        out_patch = out_patch.reshape(B, L, C, H, W)
        
        # Compute Gating Weights
        gate_input = x_noisy.reshape(B * L, C, H, W)
        gate_weights = self.gate_network(gate_input)
        gate_weights = gate_weights.reshape(B, L, 2)
        
        gate_factorized = gate_weights[:, :, 0:1].unsqueeze(-1).unsqueeze(-1)
        gate_patch = gate_weights[:, :, 1:2].unsqueeze(-1).unsqueeze(-1)
        
        # Fuse Pathways
        out_weighted_factorized = out_factorized * gate_factorized
        out_weighted_patch = out_patch * gate_patch
        
        out_concat = torch.cat([out_weighted_factorized, out_weighted_patch], dim=2)
        out_concat_flat = out_concat.reshape(B * L, C * 2, H, W)
        out_fused = self.output_fusion(out_concat_flat)
        out_fused = out_fused.reshape(B, L, C, H, W)
        
        return x_noisy + self.output_scale * out_fused
    
    def get_gate_statistics(self, x_noisy):
        """Utility to analyze gate behavior during training/inference."""
        B, L, C, H, W = x_noisy.shape
        
        with torch.no_grad():
            gate_input = x_noisy.reshape(B * L, C, H, W)
            gate_weights = self.gate_network(gate_input).reshape(B, L, 2)
            
            return {
                'factorized_mean': gate_weights[:, :, 0].mean().item(),
                'factorized_std': gate_weights[:, :, 0].std().item(),
                'patch_mean': gate_weights[:, :, 1].mean().item(),
                'patch_std': gate_weights[:, :, 1].std().item(),
                'gate_weights': gate_weights
            }


# ============================================================================
# LOW-RES CONTEXT ENCODER
# ============================================================================

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
    Provides richer context for conditioning at higher resolutions.
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
    
    @torch._dynamo.disable()
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


# ============================================================================
# FORWARD DIFFUSION
# ============================================================================

class ForwardDiffusion():
    def __init__(self, timesteps=250, beta_start=1e-4, beta_end=0.02, schedule='cosine'):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule = schedule
        
        if schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(timesteps)
        elif schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        self.alphas = 1.0 - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alpha_hats = torch.sqrt(self.alpha_hats)
        self.sqrt_one_minus_alpha_hats = torch.sqrt(1 - self.alpha_hats)

        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_hats = self.alpha_hats.to(device)
        self.sqrt_alpha_hats = self.sqrt_alpha_hats.to(device)
        self.sqrt_one_minus_alpha_hats = self.sqrt_one_minus_alpha_hats.to(device)
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps)
        
        alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999)
        
        return betas

    def forward(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0).to(dtype=MASTER_DTYPE)

        sqrt_alpha_hat = self.sqrt_alpha_hats[t].view(-1, 1, 1, 1).to(x0.device)
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hats[t].view(-1, 1, 1, 1).to(x0.device)
        
        xt = sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * noise
        return xt, noise


# ============================================================================
# MAIN MODEL V5 WITH HYBRID ATTENTION
# ============================================================================

class LayerXLayerDiffusionModelV5(nn.Module):
    """
    Layer-by-Layer 3D Diffusion Model with Hybrid Causal Attention.
    
    Key improvements over V4:
    - Hybrid attention combining factorized + patch-based approaches
    - Richer input encoding
    - Gate monitoring for training analysis
    """
    def __init__(
        self,
        layer_context_dim=64,
        granularity=128,
        text_context_dim=768,
        max_context_layers=None,
        in_channels=1,
        model_channels=128,
        context_dim=512,
        attention_resolutions=[8, 16],
        use_low_res_context=False,
        current_low_res_layer_context=False,
        hybrid_num_heads=8,
        hybrid_patch_size=4,
        hybrid_patch_downsample=2,
        hybrid_dropout=0.1,
    ):
        super().__init__()
        
        self.granularity = granularity
        self.model_channels = model_channels
        self.layer_context_dim = layer_context_dim
        self.text_context_dim = text_context_dim
        self.max_context_layers = max_context_layers
        self.use_low_res_context = use_low_res_context
        self.current_low_res_layer_context = current_low_res_layer_context
        
        # Layer Positional Embeddings
        self.layer_pos_emb = SinusoidalPosEmb(layer_context_dim)
        
        self.layer_context_conditioning_mlp = nn.Sequential(
            nn.Linear(layer_context_dim + text_context_dim, layer_context_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(layer_context_dim * 2, text_context_dim)
        )
        
        # Hybrid Causal Attention
        self.hybrid_layer_attn = HybridCausalAttention(
            channels=model_channels,
            num_heads=hybrid_num_heads,
            patch_size=hybrid_patch_size,
            dropout=hybrid_dropout,
            patch_downsample=hybrid_patch_downsample
        )
        
        # Time Embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Enhanced Input Encoder
        self.input_encoder = nn.Sequential(
            nn.Conv2d(in_channels, model_channels // 2, 3, padding=1),
            nn.GroupNorm(8, model_channels // 2),
            nn.SiLU(),
            nn.Conv2d(model_channels // 2, model_channels // 2, 3, padding=1),
            nn.GroupNorm(8, model_channels // 2),
            nn.SiLU(),
            nn.Conv2d(model_channels // 2, model_channels, 3, padding=1),
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
        )
        
        # Low-Res Context (Optional)
        if use_low_res_context:
            self.low_res_encoder = LowResContextEncoder(
                in_channels=1,
                base_channels=64,
                mode='spatial_aware',
            )
            self.low_res_cross_attn = MultiHeadCrossAttention(
                channels=model_channels,
                context_dim=128,
                num_heads=4
            )
        
        # UNet Down Blocks
        self.down_block1 = ResnetBlockWithAttention(
            model_channels, model_channels * 2, time_embed_dim, context_dim,
            layer_context_dim=layer_context_dim,
            use_attention=(32 in attention_resolutions),
            num_heads=8
        )
        self.down_block2 = ResnetBlockWithAttention(
            model_channels * 2, model_channels * 4, time_embed_dim, context_dim,
            layer_context_dim=layer_context_dim,
            use_attention=(16 in attention_resolutions),
            num_heads=8
        )
        self.down_block3 = ResnetBlockWithAttention(
            model_channels * 4, model_channels * 4, time_embed_dim, context_dim,
            layer_context_dim=layer_context_dim,
            use_attention=(8 in attention_resolutions),
            num_heads=8
        )
        
        self.downsample1 = nn.MaxPool2d(2)
        self.downsample2 = nn.MaxPool2d(2)
        
        # UNet Middle Block
        self.mid_block = ResnetBlockWithAttention(
            model_channels * 4, model_channels * 4, time_embed_dim, context_dim,
            layer_context_dim=layer_context_dim,
            use_attention=True,
            num_heads=8
        )
        
        # UNet Up Blocks
        self.up_block1 = ResnetBlockWithAttention(
            model_channels * 4, model_channels * 4, time_embed_dim, context_dim,
            layer_context_dim=layer_context_dim,
            use_attention=(8 in attention_resolutions),
            num_heads=8
        )
        self.up_block2 = ResnetBlockWithAttention(
            model_channels * 4 + model_channels * 4, model_channels * 2, time_embed_dim, context_dim,
            layer_context_dim=layer_context_dim,
            use_attention=(16 in attention_resolutions),
            num_heads=8
        )
        self.up_block3 = ResnetBlockWithAttention(
            model_channels * 2 + model_channels * 2, model_channels, time_embed_dim, context_dim,
            layer_context_dim=layer_context_dim,
            use_attention=(32 in attention_resolutions),
            num_heads=8
        )
        
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Output
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, in_channels, 3, padding=1)
        )

    def forward(self, x, t, l, context, prev_layers=None, low_res_features=None):
        """Single-layer forward pass (for inference/sequential generation)."""
        B = x.shape[0]
        model_dtype = next(self.parameters()).dtype
        
        l = l.to(model_dtype)
        t = t.to(model_dtype)
        x = x.to(model_dtype)
        context = context.to(model_dtype)
        
        layer_pos = self.layer_pos_emb(l)
        
        layer_pos_expanded = layer_pos.unsqueeze(1).expand(-1, context.shape[1], -1)
        combined_context = torch.cat([layer_pos_expanded, context], dim=-1)
        context = self.layer_context_conditioning_mlp(combined_context)
        
        h = self.input_encoder(x)
        
        if low_res_features is not None and self.use_low_res_context:
            if self.current_low_res_layer_context:
                batch_size = x.shape[0]
                layer_indices = l.long()
                current_low_res_feat = low_res_features[
                    torch.arange(batch_size, device=x.device),
                    layer_indices
                ].unsqueeze(1)
                h = self.low_res_cross_attn(h, current_low_res_feat)
            else:
                h = self.low_res_cross_attn(h, low_res_features)
        
        if prev_layers is not None and prev_layers.numel() > 0:
            N = prev_layers.shape[1]
            
            prev_flat = prev_layers.reshape(B * N, prev_layers.shape[2],
                                           prev_layers.shape[3], prev_layers.shape[4])
            prev_encoded = self.input_encoder(prev_flat.to(model_dtype))
            prev_encoded = prev_encoded.reshape(B, N, self.model_channels,
                                               prev_encoded.shape[2], prev_encoded.shape[3])
            
            h_expanded = h.unsqueeze(1)
            
            all_layers_noisy = torch.cat([prev_encoded, h_expanded], dim=1)
            all_layers_clean = torch.cat([prev_encoded, h_expanded], dim=1)
            
            all_layers_attended = self.hybrid_layer_attn(all_layers_noisy, all_layers_clean)
            
            h = all_layers_attended[:, -1]
        
        time_emb = self.time_embed(t)
        
        h1 = self.down_block1(h, time_emb, context, layer_pos)
        h = self.downsample1(h1)
        h2 = self.down_block2(h, time_emb, context, layer_pos)
        h = self.downsample2(h2)
        h3 = self.down_block3(h, time_emb, context, layer_pos)
        
        h = self.mid_block(h3, time_emb, context, layer_pos)
        
        h = self.up_block1(h, time_emb, context, layer_pos)
        h = self.upsample1(h)
        h = torch.cat([h, h2], dim=1)
        h = self.up_block2(h, time_emb, context, layer_pos)
        h = self.upsample2(h)
        h = torch.cat([h, h1], dim=1)
        h = self.up_block3(h, time_emb, context, layer_pos)
        
        return self.output_conv(h)

    def forward_parallel(self, x_noisy_all, t_all, context, x_clean_all, low_res_features=None):
        """
        Parallel forward pass for all layers with hybrid causal attention.
        Used during training for efficiency.
        """
        B, L, C, H, W = x_noisy_all.shape
        device = x_noisy_all.device
        model_dtype = next(self.parameters()).dtype
        
        # Layer Positional Embeddings
        layer_indices = torch.arange(L, device=device, dtype=model_dtype)
        layer_pos_all = self.layer_pos_emb(layer_indices)
        layer_pos_all = layer_pos_all.unsqueeze(0).expand(B, -1, -1)
        
        # Context Processing
        context_expanded = context.unsqueeze(1).expand(-1, L, -1, -1)
        
        layer_pos_for_context = layer_pos_all.unsqueeze(2).expand(-1, -1, context.shape[1], -1)
        combined_context = torch.cat([layer_pos_for_context, context_expanded], dim=-1)
        combined_context_flat = combined_context.reshape(B * L, context.shape[1], -1)
        context_processed = self.layer_context_conditioning_mlp(combined_context_flat)
        
        # Encode All Layers
        x_noisy_flat = x_noisy_all.reshape(B * L, C, H, W)
        x_clean_flat = x_clean_all.reshape(B * L, C, H, W)
        
        h_noisy = self.input_encoder(x_noisy_flat)
        h_clean = self.input_encoder(x_clean_flat)
        
        h_noisy = h_noisy.reshape(B, L, self.model_channels, H, W)
        h_clean = h_clean.reshape(B, L, self.model_channels, H, W)
        
        # Hybrid Causal Attention
        h = self.hybrid_layer_attn(h_noisy, h_clean)
        
        # Low-Res Conditioning (Optional)
        if low_res_features is not None and self.use_low_res_context:
            h_flat = h.reshape(B * L, self.model_channels, H, W)
            if self.current_low_res_layer_context:
                low_res_flat = low_res_features.reshape(B * L, 1, -1)
            else:
                low_res_expanded = low_res_features.unsqueeze(1).expand(-1, L, -1, -1)
                low_res_flat = low_res_expanded.reshape(B * L, L, -1)
            
            h_flat = self.low_res_cross_attn(h_flat, low_res_flat)
            h = h_flat.reshape(B, L, self.model_channels, H, W)
        
        # UNet Processing (All Layers in Parallel)
        h = h.reshape(B * L, self.model_channels, H, W)
        
        t_flat = t_all.reshape(B * L).to(model_dtype)
        time_emb = self.time_embed(t_flat)
        
        layer_pos_flat = layer_pos_all.reshape(B * L, -1)
        
        h1 = self.down_block1(h, time_emb, context_processed, layer_pos_flat)
        h = self.downsample1(h1)
        h2 = self.down_block2(h, time_emb, context_processed, layer_pos_flat)
        h = self.downsample2(h2)
        h3 = self.down_block3(h, time_emb, context_processed, layer_pos_flat)
        
        h = self.mid_block(h3, time_emb, context_processed, layer_pos_flat)
        
        h = self.up_block1(h, time_emb, context_processed, layer_pos_flat)
        h = self.upsample1(h)
        h = torch.cat([h, h2], dim=1)
        h = self.up_block2(h, time_emb, context_processed, layer_pos_flat)
        h = self.upsample2(h)
        h = torch.cat([h, h1], dim=1)
        h = self.up_block3(h, time_emb, context_processed, layer_pos_flat)
        
        out = self.output_conv(h)
        out = out.reshape(B, L, C, H, W)
        
        return out


# ============================================================================
# TRAINER V5
# ============================================================================

class LayerXLayerDiffusionTrainerV5:
    """
    Trainer for LayerXLayerDiffusionModelV5 with hybrid attention.
    Adds gate monitoring and attention analysis capabilities.
    """
    def __init__(self, model, diffusion, scheduler=None,
                 teacher_forcing=True, use_ddim=True, ddim_steps=50, ddim_eta=0.0,
                 low_res_contexts=None, parallel_training=True):
        self.model = model
        self.diffusion = diffusion
        lr = args.learning_rate if args.learning_rate is not None else 1e-5
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = scheduler
        self.teacher_forcing = teacher_forcing
        self.use_ddim = use_ddim
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        self.low_res_contexts = low_res_contexts or {}
        self.parallel_training = parallel_training
        
        self.gate_history = []

    def compute_loss(self, x0, context, low_res_context=None, backward=True):
        """Unified loss computation."""
        return self.compute_loss_parallel(x0, context, low_res_context, backward)

    def compute_loss_parallel(self, x0, context, low_res_context=None, backward=True):
        """Parallel training loss with gate monitoring."""
        if backward:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()
        
        B, C, H, W, D = x0.shape
        device = x0.device
        model_dtype = next(self.model.parameters()).dtype
        L = D
        
        all_layers = x0.permute(0, 4, 1, 2, 3).to(model_dtype)
        
        t = torch.randint(0, self.diffusion.timesteps, (B,), device=device).long()
        t_all = t.unsqueeze(1).expand(-1, L)
        
        noise = torch.randn_like(all_layers)
        sqrt_alpha = self.diffusion.sqrt_alpha_hats[t].view(B, 1, 1, 1, 1).to(model_dtype)
        sqrt_one_minus = self.diffusion.sqrt_one_minus_alpha_hats[t].view(B, 1, 1, 1, 1).to(model_dtype)
        x_noisy = sqrt_alpha * all_layers + sqrt_one_minus * noise
        
        low_res_features = None
        if low_res_context is not None and self.model.use_low_res_context:
            with torch.no_grad():
                low_res_features = self.model.low_res_encoder(
                    low_res_context.to(model_dtype), L
                ).detach()
        
        predicted_noise = self.model.forward_parallel(
            x_noisy,
            t_all,
            context.to(model_dtype),
            all_layers,
            low_res_features=low_res_features
        )
        
        loss = F.mse_loss(predicted_noise.float(), noise.float())
        
        # Monitor gate statistics periodically
        if backward and hasattr(self.model, 'hybrid_layer_attn'):
            if len(self.gate_history) % 100 == 0:
                with torch.no_grad():
                    x_noisy_flat = x_noisy.reshape(B * L, C, H, W)
                    h_noisy = self.model.input_encoder(x_noisy_flat)
                    h_noisy = h_noisy.reshape(B, L, self.model.model_channels, H, W)
                    
                    gate_stats = self.model.hybrid_layer_attn.get_gate_statistics(h_noisy)
                    self.gate_history.append({
                        'step': len(self.gate_history),
                        'factorized_mean': gate_stats['factorized_mean'],
                        'patch_mean': gate_stats['patch_mean'],
                    })
        
        if backward:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
        
        return loss.item()

    def train_step(self, x0, context, low_res_context=None):
        """Single training step."""
        return self.compute_loss_parallel(x0, context, low_res_context, backward=True)

    def get_gate_analysis(self):
        """Return gate statistics for analysis."""
        if not self.gate_history:
            return None
        
        factorized_means = [g['factorized_mean'] for g in self.gate_history]
        patch_means = [g['patch_mean'] for g in self.gate_history]
        
        return {
            'history': self.gate_history,
            'factorized_trend': factorized_means,
            'patch_trend': patch_means,
            'final_factorized': factorized_means[-1] if factorized_means else None,
            'final_patch': patch_means[-1] if patch_means else None,
        }

    def sample(self, context, shape, device):
        """Generate samples using DDIM or full diffusion."""
        if self.use_ddim:
            return self.ddim_sample(context, shape, device, self.ddim_steps, self.ddim_eta)
        else:
            return self.full_diffusion_sample(context, shape, device)

    def ddim_sample(self, context, shape, device, ddim_steps=50, eta=0.0):
        """DDIM sampling - sequential layer-by-layer."""
        self.model.eval()
        model_dtype = next(self.model.parameters()).dtype
        
        with torch.no_grad():
            batch_size = shape[0]
            voxel_grid = torch.zeros(
                (batch_size, shape[1], shape[2], shape[3], self.model.granularity),
                device=device, dtype=model_dtype
            )
            
            timestep_interval = self.diffusion.timesteps // ddim_steps
            ddim_timesteps = list(range(0, self.diffusion.timesteps, timestep_interval))
            ddim_timesteps.reverse()
            
            generated_layers = []
            
            for layer_idx in range(self.model.granularity):
                l = torch.full((batch_size,), layer_idx, device=device, dtype=model_dtype)
                x = torch.randn(shape, device=device, dtype=model_dtype)
                
                if len(generated_layers) > 0:
                    prev_layers = torch.stack(generated_layers, dim=1)
                else:
                    prev_layers = None
                
                for i, t in enumerate(ddim_timesteps):
                    t_batch = torch.full((batch_size,), t, device=device, dtype=model_dtype)
                    predicted_noise = self.model(x, t_batch, l, context, prev_layers)
                    
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

    def full_diffusion_sample(self, context, shape, device):
        """Full diffusion sampling - sequential layer-by-layer."""
        self.model.eval()
        model_dtype = next(self.model.parameters()).dtype
        
        with torch.no_grad():
            batch_size = shape[0]
            voxel_grid = torch.zeros(
                (batch_size, shape[1], shape[2], shape[3], self.model.granularity),
                device=device, dtype=model_dtype
            )
            
            generated_layers = []
            
            for layer_idx in range(self.model.granularity):
                l = torch.full((batch_size,), layer_idx, device=device, dtype=model_dtype)
                x = torch.randn(shape, device=device, dtype=model_dtype)
                
                if len(generated_layers) > 0:
                    prev_layers = torch.stack(generated_layers, dim=1)
                else:
                    prev_layers = None
                
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
                    
                    x = (1 / torch.sqrt(alpha_t)) * (
                        x - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * predicted_noise
                    ) + torch.sqrt(beta_t) * noise
                
                voxel_grid[:, :, :, :, layer_idx] = x
                generated_layers.append(x.detach())
            
            return voxel_grid.clamp(0, 1)


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_model_v5(granularity, use_low_res_context, args, device):
    """
    Factory function to create LayerXLayerDiffusionModelV5 with appropriate config.
    """
    if granularity <= 16:
        patch_size = 2
        patch_downsample = 1
    elif granularity <= 32:
        patch_size = 4
        patch_downsample = 2
    else:
        patch_size = 4
        patch_downsample = 4
    
    model = LayerXLayerDiffusionModelV5(
        layer_context_dim=64,
        granularity=granularity,
        text_context_dim=768,
        max_context_layers=16,
        in_channels=1,
        model_channels=512,
        context_dim=768,
        attention_resolutions=[8, 16, 32],
        use_low_res_context=use_low_res_context,
        current_low_res_layer_context=args.current_layer_only,
        hybrid_num_heads=args.hybrid_num_heads,
        hybrid_patch_size=patch_size,
        hybrid_patch_downsample=patch_downsample,
        hybrid_dropout=args.hybrid_dropout,
    )
    
    return model


# ============================================================================
# DATASET
# ============================================================================

class VoxelDataset(Dataset):
    def __init__(self, npy_folder_path, description_folder_path, transform=None, 
                 granularity=128, test_count=0, enable_fuzzy_matching=False):
        self.npy_folder_path = Path(npy_folder_path)
        self.description_folder_path = Path(description_folder_path)
        self.transform = transform
        self.granularity = granularity
        self.default_description = [
            "A 3D model of an object.",
            "A geometric shape in 3D space.",
            "A voxel representation of a structure.",
            "A 3D voxel grid of an item.",
            "A digital 3D model.",
            "A voxel-based 3D object.",
            "A 3D shape made of voxels.",
        ]
        
        self.enable_fuzzy_matching = enable_fuzzy_matching
        self.description_misses = []
        
        print("Loading file lists...")
        
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
        
        if self.enable_fuzzy_matching:
            self.available_descriptions = {
                entry.name[:-4]
                for entry in os.scandir(description_folder_path)
                if entry.name.endswith('.txt') and entry.is_file()
            }
            print(f"Found {len(self.available_descriptions)} description files (fuzzy matching enabled)")
        else:
            self.available_descriptions = None
            print("Fuzzy matching disabled - descriptions loaded on-demand only")
        
        print(f"Found {len(self.file_list)} .npy files in {npy_folder_path}")
        print(f"Voxel grid granularity: {self.granularity}x{self.granularity}x{self.granularity}")
    
    @lru_cache(maxsize=10000)
    def _load_description(self, filename_stem):
        desc_file = self.description_folder_path / f"{filename_stem}.txt"
        
        if desc_file.exists():
            try:
                with open(desc_file, 'r') as f:
                    return f.read().strip()
            except Exception as e:
                print(f"Error loading {desc_file}: {e}")
                return random.choice(self.default_description)
        else:
            if self.enable_fuzzy_matching and self.available_descriptions is not None:
                closest_matches = get_close_matches(filename_stem, self.available_descriptions, n=1, cutoff=0.6)
                closest_match = closest_matches[0] if closest_matches else "No close match found"
                
                self.description_misses.append({
                    'missing': filename_stem,
                    'closest_match': closest_match
                })
                
                if closest_matches:
                    print(f"Fuzzy matched: {filename_stem} -> {closest_match}")
                    matched_file = self.description_folder_path / f"{closest_match}.txt"
                    try:
                        with open(matched_file, 'r') as f:
                            return f.read().strip()
                    except Exception:
                        pass
            else:
                self.description_misses.append({
                    'missing': filename_stem,
                    'closest_match': 'N/A (fuzzy matching disabled)'
                })
            
            return random.choice(self.default_description)

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        voxel_grid = np.load(self.file_list[idx]).astype(np.float16)
        assert voxel_grid.shape == (self.granularity, self.granularity, self.granularity)
        
        if self.transform:
            voxel_grid = self.transform(voxel_grid)
        
        voxel_grid = torch.from_numpy(voxel_grid).unsqueeze(0)
        
        filename_key = self.file_list[idx].stem
        description = self._load_description(filename_key)
        
        return voxel_grid, description, filename_key
    
    def save_miss_report(self, output_path="description_misses.txt"):
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


# ============================================================================
# TRAINING VISUALIZER
# ============================================================================

class TrainingVoxelVisualizer:
    def __init__(self, save_dir, max_voxels=30000):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_voxels = max_voxels
    
    def visualize_single_voxel(self, voxel_grid, ax, threshold=0.5, view_angle=(30, 45), colormap='viridis'):
        filled = voxel_grid > threshold
        colors = plt.cm.get_cmap(colormap)(voxel_grid)
        
        ax.voxels(filled, facecolors=colors, edgecolors='gray', alpha=0.8, linewidth=0.1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        elev, azim = view_angle
        ax.view_init(elev=elev, azim=azim)
        
        max_range = voxel_grid.shape[0]
        ax.set_xlim([0, max_range])
        ax.set_ylim([0, max_range])
        ax.set_zlim([0, max_range])
        
        return filled.sum()
    
    def visualize_epoch_samples(self, trainer, epoch, descriptions=None,
                                embedding_cache=None, context_encoder=None, 
                                tokenizer=None, granularity=32, device='cuda', 
                                num_samples=4, view_angles=[(30, 45), (60, 120)],
                                colormap='viridis', show_slices=True, show_progress=False):
        if show_progress:
            print(f"\n{'='*60}")
            print(f"GENERATING VISUALIZATION FOR EPOCH {epoch}")
            print(f"{'='*60}")
        
        if descriptions is None:
            descriptions = [
                "A simple cube",
                "A sphere",
                "A pyramid",
                "An abstract shape",
            ][:num_samples]
        
        with torch.no_grad():
            if embedding_cache is not None:
                try:
                    context = embedding_cache.get_batch_embeddings(descriptions, device)
                except KeyError:
                    if context_encoder is not None and tokenizer is not None:
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
                        context = context.to(device)
                    else:
                        context = torch.randn(num_samples, 77, 768).to(device)
            elif context_encoder is not None and tokenizer is not None:
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
                context = context.to(device)
            else:
                context = torch.randn(num_samples, 77, 768).to(device)
            
            samples = trainer.sample(
                context,
                shape=(num_samples, 1, granularity, granularity),
                device=device
            )
            
            samples = samples.clamp(0, 1).cpu().numpy()
        
        num_views = len(view_angles)
        
        if show_slices:
            fig = plt.figure(figsize=(5 * (num_views + 1), 4 * num_samples))
            
            for i in range(num_samples):
                voxel_grid = samples[i, 0]
                
                for j, (elev, azim) in enumerate(view_angles):
                    ax = fig.add_subplot(num_samples, num_views + 1, 
                                        i * (num_views + 1) + j + 1, 
                                        projection='3d')
                    
                    num_voxels = self.visualize_single_voxel(
                        voxel_grid, ax, view_angle=(elev, azim),
                        colormap=colormap
                    )
                    
                    if j == 0:
                        desc_short = (descriptions[i][:30] + '...') if len(descriptions[i]) > 30 else descriptions[i]
                        ax.set_title(f'{desc_short}\n{num_voxels:,} voxels', fontsize=10)
                
                ax_slice = fig.add_subplot(num_samples, num_views + 1, 
                                          i * (num_views + 1) + num_views + 1)
                mid_slice = voxel_grid[:, :, granularity // 2]
                ax_slice.imshow(mid_slice, cmap='gray', vmin=0, vmax=1)
                ax_slice.set_title(f'Middle Slice (z={granularity//2})', fontsize=10)
                ax_slice.axis('off')
        else:
            fig = plt.figure(figsize=(5 * num_views, 4 * num_samples))
            
            for i in range(num_samples):
                voxel_grid = samples[i, 0]
                
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
        
        sampling_method = "DDIM" if trainer.use_ddim else "Full Diffusion"
        plt.suptitle(f'Epoch {epoch} - Generated Voxel Samples ({sampling_method})', 
                    fontsize=14, y=0.995)
        plt.tight_layout()
        
        save_path = self.save_dir / f'epoch_{epoch:04d}_samples.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show_progress:
            print(f"✓ Visualization saved to: {save_path}")
        
        plt.show()
        plt.close()
        
        return samples


# ============================================================================
# PROGRESSIVE GRANULARITY TRAINER
# ============================================================================

def create_distributed_dataloaders(train_dataset, val_dataset, batch_size):
    """Create DataLoaders with distributed sampling if in distributed mode."""
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=42
        )
        
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        
        shuffle_train = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True
    
    def collate_fn(batch):
        voxels = torch.stack([item[0] for item in batch])
        descriptions = [item[1] for item in batch]
        file_stems = [item[2] for item in batch]
        return voxels, descriptions, file_stems
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=shuffle_train if train_sampler is None else False,
        num_workers=4 * (world_size if is_distributed else 1),
        pin_memory=True,
        prefetch_factor=4,
        drop_last=True,
        persistent_workers=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=4 * (world_size if is_distributed else 1),
        pin_memory=True,
        drop_last=False,
        prefetch_factor=4,
        persistent_workers=True,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, train_sampler


class FSDPProgressiveGranularityTrainer:
    """FSDP-compatible progressive training manager using V5 model."""
    
    def __init__(self, granularities=[16, 32, 64], base_config=None):
        self.granularities = granularities
        self.base_config = base_config or {}
        self.stage_checkpoints = {}
        self.low_res_cache = {}

        self.visualizer = TrainingVoxelVisualizer(
            save_dir='./training_visualizations'
        )

        self.plots_dir = Path('./training_plots')
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_history = {
            'stages': {},
            'global_train_loss': [],
            'global_val_loss': [],
            'global_epochs': [],
        }

    def plot_loss_curves(self, stage_idx, stage_metrics, global_epoch_offset, current_epoch):
        """Plot and save loss curves."""
        if rank != 0:
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Training Progress - Stage {stage_idx + 1} (Epoch {current_epoch})', 
                        fontsize=14, fontweight='bold')
            
            ax1 = axes[0, 0]
            if stage_metrics['train_loss']:
                epochs = list(range(1, len(stage_metrics['train_loss']) + 1))
                ax1.plot(epochs, stage_metrics['train_loss'], 'b-', linewidth=2, alpha=0.7)
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Training Loss')
                ax1.set_title(f'Stage {stage_idx + 1} Training Loss ({stage_metrics["granularity"]}³)')
                ax1.grid(True, alpha=0.3)
            
            ax2 = axes[0, 1]
            if stage_metrics['val_loss']:
                val_epochs = list(range(args.validation_interval, 
                                       len(stage_metrics['train_loss']) + 1, 
                                       args.validation_interval))[:len(stage_metrics['val_loss'])]
                ax2.plot(val_epochs, stage_metrics['val_loss'], 'r-o', linewidth=2, alpha=0.7)
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Validation Loss')
                ax2.set_title(f'Stage {stage_idx + 1} Validation Loss')
                ax2.grid(True, alpha=0.3)
                
                if stage_metrics['val_loss']:
                    best_val = min(stage_metrics['val_loss'])
                    best_idx = stage_metrics['val_loss'].index(best_val)
                    best_epoch = val_epochs[best_idx]
                    ax2.scatter([best_epoch], [best_val], c='gold', s=200, marker='*', 
                               zorder=5, label=f'Best: {best_val:.4f}')
                    ax2.legend()
            
            ax3 = axes[1, 0]
            if self.metrics_history['global_train_loss']:
                global_epochs = list(range(1, len(self.metrics_history['global_train_loss']) + 1))
                ax3.plot(global_epochs, self.metrics_history['global_train_loss'], 
                        'g-', linewidth=2, alpha=0.7)
                ax3.set_xlabel('Global Epoch')
                ax3.set_ylabel('Training Loss')
                ax3.set_title('Global Training Loss')
                ax3.grid(True, alpha=0.3)
            
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            summary_text = "Current Training Status\n" + "="*40 + "\n\n"
            summary_text += f"Stage: {stage_idx + 1}/{len(self.granularities)}\n"
            summary_text += f"Granularity: {stage_metrics['granularity']}³\n"
            summary_text += f"Epoch: {current_epoch}\n\n"
            
            if stage_metrics['train_loss']:
                summary_text += f"Current Train Loss: {stage_metrics['train_loss'][-1]:.4f}\n"
            
            if stage_metrics['val_loss']:
                summary_text += f"Latest Val Loss: {stage_metrics['val_loss'][-1]:.4f}\n"
                summary_text += f"Best Val Loss: {min(stage_metrics['val_loss']):.4f}\n"
            
            ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
            
            plt.tight_layout()
            
            save_path = self.plots_dir / f'stage{stage_idx}_epoch{current_epoch}.png'
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print_main(f"⚠ Error plotting loss curves: {e}")

    def train_all_stages(self, epochs_per_stage, device, embedding_cache):
        """Train all granularity stages sequentially."""
        print_main("\n" + "="*60)
        print_main("PROGRESSIVE GRANULARITY TRAINING (V5 - Hybrid Attention)")
        print_main("="*60)
        print_main(f"Stages: {self.granularities}")
        print_main(f"Epochs per stage: {epochs_per_stage}")
        print_main("="*60 + "\n")
        
        for stage_idx in range(len(self.granularities)):
            granularity = self.granularities[stage_idx]
            epochs = epochs_per_stage[stage_idx]
            
            print_main(f"\nCreating datasets for {granularity}³ resolution...")
            stage_train_dataset = VoxelDataset(
                npy_folder_path=f'/home/ad.msoe.edu/benzshawelt/Kedziora/kedziora_research/description_data_generation/objaverse_parallel/objaverse_voxels/{granularity}',
                description_folder_path='/home/ad.msoe.edu/benzshawelt/Kedziora/kedziora_research/description_data_generation/objaverse_parallel/objaverse_descriptions',
                granularity=granularity,
                test_count=0,
                enable_fuzzy_matching=args.enable_fuzzy_matching
            )
            
            stage_val_dataset = VoxelDataset(
                npy_folder_path=f'/home/ad.msoe.edu/benzshawelt/Kedziora/kedziora_research/description_data_generation/objaverse_parallel/objaverse_voxels/{granularity}',
                description_folder_path='/home/ad.msoe.edu/benzshawelt/Kedziora/kedziora_research/description_data_generation/objaverse_parallel/objaverse_descriptions',
                granularity=granularity,
                test_count=400,
                enable_fuzzy_matching=args.enable_fuzzy_matching
            )
            
            stage_metrics = self.train_stage(
                stage_idx=stage_idx,
                train_dataset=stage_train_dataset,
                val_dataset=stage_val_dataset,
                epochs=epochs,
                device=device,
                embedding_cache=embedding_cache
            )
            
            global_epoch_offset = sum(epochs_per_stage[:stage_idx])
            for epoch_loss in stage_metrics['train_loss']:
                self.metrics_history['global_train_loss'].append(epoch_loss)
            
            for val_loss in stage_metrics['val_loss']:
                self.metrics_history['global_val_loss'].append(val_loss)
            
            stage_epoch_nums = [global_epoch_offset + e for e in stage_metrics['epochs']]
            self.metrics_history['global_epochs'].extend(stage_epoch_nums)
        
        print_main("\n" + "="*60)
        print_main("PROGRESSIVE TRAINING COMPLETE!")
        print_main("="*60)
        
        return {
            'metrics_history': self.metrics_history,
            'checkpoints': self.stage_checkpoints,
            'granularities': self.granularities
        }

    def save_stage_checkpoint(self, model, stage_idx, device):
        """Save FSDP model state."""
        granularity = self.granularities[stage_idx]
        
        if args.shard_model:
            save_policy = FullStateDictConfig(
                offload_to_cpu=True, 
                rank0_only=True
            )
            with FSDP.state_dict_type(
                model, 
                StateDictType.FULL_STATE_DICT, 
                save_policy
            ):
                if rank == 0:
                    state_dict = model.state_dict()
                    checkpoint_path = f'stage{stage_idx}_gran{granularity}_v5.pth'
                    torch.save({
                        'model_state_dict': state_dict,
                        'granularity': granularity,
                        'stage_idx': stage_idx,
                        'model_version': 'v5'
                    }, checkpoint_path)
                    self.stage_checkpoints[stage_idx] = checkpoint_path
                    print(f"✓ Stage {stage_idx} checkpoint saved: {checkpoint_path}")
        else:
            checkpoint_path = f'stage{stage_idx}_gran{granularity}_v5.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'granularity': granularity,
                'stage_idx': stage_idx,
                'model_version': 'v5'
            }, checkpoint_path)
            self.stage_checkpoints[stage_idx] = checkpoint_path
        
        if is_distributed:
            dist.barrier()
            if rank == 0:
                path_list = [self.stage_checkpoints[stage_idx]]
            else:
                path_list = [None]
            dist.broadcast_object_list(path_list, src=0)
            self.stage_checkpoints[stage_idx] = path_list[0]

    def load_prev_stage_model_gpu(self, stage_idx, device):
        """Load previous stage model on GPU."""
        if stage_idx == 0 or stage_idx - 1 not in self.stage_checkpoints:
            return None
        
        checkpoint_path = self.stage_checkpoints[stage_idx - 1]
        prev_granularity = self.granularities[stage_idx - 1]
        
        use_low_res_prev = stage_idx - 1 > 0

        prev_model = create_model_v5(prev_granularity, use_low_res_prev, args, device)
        prev_model = prev_model.to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
        prev_model.load_state_dict(state_dict)
        prev_model.eval()
        
        print(f"✓ Loaded V5 stage {stage_idx-1} model for stage {stage_idx}")

        return prev_model

    def precompute_low_res_contexts(self, stage_idx, train_dataset, device, 
                                    embedding_cache, batch_size=4):
        """Precompute low-res contexts before training starts."""
        if stage_idx == 0:
            return {}
        
        print_main(f"\nPrecomputing low-res contexts for stage {stage_idx}...")
        
        cache_path = f'low_res_context_stage{stage_idx}_v5.pth'
        if Path(cache_path).exists():
            print_main(f"✓ Loading existing cache from {cache_path}")
            low_res_contexts = torch.load(cache_path, map_location='cpu')
            print_main(f"✓ Loaded {len(low_res_contexts)} cached contexts")
            return low_res_contexts
        
        prev_model = self.load_prev_stage_model_gpu(stage_idx, device)
        
        prev_low_res_contexts = {}
        if stage_idx > 1:
            prev_context_path = f'low_res_context_stage{stage_idx - 1}_v5.pth'
            if Path(prev_context_path).exists():
                print_main(f"  Loading recursive contexts from {prev_context_path}...")
                prev_low_res_contexts = torch.load(prev_context_path, map_location='cpu')
        
        low_res_contexts = {}
        curr_granularity = self.granularities[stage_idx]
        prev_granularity = self.granularities[stage_idx - 1]
        
        if rank == 0:
            curr_stems = set()
            for file_path in train_dataset.file_list:
                curr_stems.add(file_path.stem)
            
            prev_dataset = VoxelDataset(
                npy_folder_path=f'/home/ad.msoe.edu/benzshawelt/Kedziora/kedziora_research/description_data_generation/objaverse_parallel/objaverse_voxels/{prev_granularity}',
                description_folder_path='/home/ad.msoe.edu/benzshawelt/Kedziora/kedziora_research/description_data_generation/objaverse_parallel/objaverse_descriptions',
                granularity=prev_granularity,
                test_count=0,
                enable_fuzzy_matching=args.enable_fuzzy_matching
            )
            
            prev_file_list = [f for f in prev_dataset.file_list if f.stem in curr_stems]
            print_main(f"  Will generate contexts for {len(prev_file_list)} samples")
            
            prev_dataset.file_list = prev_file_list
            
            prev_loader = DataLoader(
                prev_dataset, 
                batch_size=batch_size, 
                shuffle=False,
                num_workers=2,
                prefetch_factor=4,
                persistent_workers=True
            )
            
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(tqdm(
                    prev_loader, desc="Generating low-res contexts"
                )):
                    x0, descriptions, file_stems = batch_data
                    
                    context = embedding_cache.get_batch_embeddings(
                        descriptions, device
                    )
                    
                    batch_prev_contexts = None
                    if prev_low_res_contexts:
                        batch_prev_contexts = []
                        for stem in file_stems:
                            if stem in prev_low_res_contexts:
                                batch_prev_contexts.append(
                                    prev_low_res_contexts[stem].to(device, dtype=MASTER_DTYPE)
                                )
                            else:
                                batch_prev_contexts.append(
                                    torch.zeros(1, prev_granularity, prev_granularity, prev_granularity,
                                               device=device, dtype=MASTER_DTYPE)
                                )
                        
                        if batch_prev_contexts:
                            batch_prev_contexts = torch.stack(batch_prev_contexts)

                    samples = self._sample_gpu(
                        prev_model, context, x0.shape, prev_granularity, device,
                        low_res_contexts=batch_prev_contexts
                    )
                    
                    upsampled = F.interpolate(
                        samples,
                        size=(curr_granularity, curr_granularity, curr_granularity),
                        mode='trilinear',
                        align_corners=False
                    )
                    
                    for i, stem in enumerate(file_stems):
                        low_res_contexts[stem] = upsampled[i].cpu()
                    
                    if (batch_idx + 1) % 10 == 0:
                        self._save_context_cache(low_res_contexts, stage_idx)
            
            print_main(f"✓ Generated {len(low_res_contexts)} low-res contexts")
            self._save_context_cache(low_res_contexts, stage_idx)
            
            del prev_model
            del prev_low_res_contexts
            torch.cuda.empty_cache()
        
        if is_distributed:
            dist.barrier()
            low_res_contexts = self._load_context_cache(stage_idx)
        
        return low_res_contexts

    def _sample_gpu(self, model, context, shape, granularity, device, low_res_contexts=None):
        """Standard sampling on GPU for context generation."""
        model.eval()
        batch_size = shape[0]
        voxel_grid = torch.zeros(
            (batch_size, 1, granularity, granularity, granularity),
            device=device,
            dtype=MASTER_DTYPE
        )
        
        generated_layers = []
        diffusion = ForwardDiffusion(timesteps=100, schedule='cosine')
        
        for layer_idx in range(granularity):
            l = torch.full((batch_size,), layer_idx, dtype=MASTER_DTYPE, device=device)
            x = torch.randn((batch_size, 1, granularity, granularity), device=device, dtype=MASTER_DTYPE)
            
            if len(generated_layers) > 0:
                prev_layers = torch.stack(generated_layers, dim=1)
            else:
                prev_layers = None
            
            for t in reversed(range(diffusion.timesteps)):
                t_batch = torch.full((batch_size,), t, dtype=MASTER_DTYPE, device=device)
                
                predicted_noise = model(
                    x, t_batch, l, context, prev_layers,
                    low_res_features=low_res_contexts
                )
                
                alpha_t = diffusion.alphas[t]
                alpha_hat_t = diffusion.alpha_hats[t]
                beta_t = diffusion.betas[t]
                
                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * predicted_noise) + torch.sqrt(beta_t) * noise
            
            voxel_grid[:, :, :, :, layer_idx] = x
            generated_layers.append(x.detach())
        
        return voxel_grid.clamp(0, 1)

    def _save_context_cache(self, contexts, stage_idx):
        cache_path = f'low_res_context_stage{stage_idx}_v5.pth'
        torch.save(contexts, cache_path)
        print(f"✓ Context cache saved: {cache_path}")
    
    def _load_context_cache(self, stage_idx):
        cache_path = f'low_res_context_stage{stage_idx}_v5.pth'
        if Path(cache_path).exists():
            return torch.load(cache_path, map_location='cpu')
        return {}

    def train_stage(self, stage_idx, train_dataset, val_dataset, epochs, device, embedding_cache):
        """Train a single stage with V5 model."""
        granularity = self.granularities[stage_idx]
        use_low_res = stage_idx > 0
        
        print_main(f"\n{'='*60}")
        print_main(f"STAGE {stage_idx + 1}: Training at {granularity}³ with Hybrid Attention (V5)")
        if use_low_res:
            prev_gran = self.granularities[stage_idx - 1]
            print_main(f"Using {prev_gran}³ as conditioning")
        print_main(f"{'='*60}\n")
        
        low_res_contexts = {}
        if use_low_res:
            low_res_contexts = self.precompute_low_res_contexts(
                stage_idx, train_dataset, device, embedding_cache
            )

        stage_metrics = {
            'granularity': granularity,
            'train_loss': [],
            'val_loss': [],
            'val_occupancy': [],
            'val_density': [],
            'epochs': list(range(1, epochs + 1)),
            'gate_stats': []
        }

        # Create V5 model
        print_main("Creating V5 model with Hybrid Causal Attention...")
        model = create_model_v5(granularity, use_low_res, args, device)
        model = initialize_model_weights(model)

        if Path('encoder_checkpoints_16_to_32/encoder_best.pth').exists():
            model = load_pretrained_encoder(
                model, 
                'encoder_checkpoints_16_to_32/encoder_best.pth'
            )

        model = model.to(dtype=MASTER_DTYPE).to(device)

        if args.shard_model:
            fsdp_config = get_fsdp_config()
            model = FSDP(model, **fsdp_config)

        model = torch.compile(model, mode='max-autotune', fullgraph=True)

        print_main("✓ V5 Model created and compiled")
        print_main(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        diffusion = ForwardDiffusion(timesteps=100, schedule='cosine')
        trainer = LayerXLayerDiffusionTrainerV5(
            model=model,
            diffusion=diffusion,
            scheduler=None,
            low_res_contexts=low_res_contexts,
            use_ddim=True,
            ddim_steps=50
        )
        
        batch_size = args.batch_size if args.batch_size is not None else 4
        train_loader, val_loader, train_sampler = create_distributed_dataloaders(
            train_dataset, val_dataset, batch_size
        )
        
        best_val_loss = float('inf')
        
        print_main(f"\nStarting training for {epochs} epochs...")
        print_main("="*60)
        
        for epoch in range(epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            
            if rank == 0:
                print(f"\n[Stage {stage_idx + 1}] Epoch {epoch + 1}/{epochs}")
                print("-" * 60)
            
            epoch_metrics = self._train_epoch(
                trainer, train_loader, embedding_cache, 
                device, epoch, stage_idx
            )
            
            epoch_loss = epoch_metrics['train_loss']
            stage_metrics['train_loss'].append(epoch_loss)
            
            if rank == 0:
                print(f"Training Loss: {epoch_loss:.4f}")
                
                # Log gate statistics
                gate_analysis = trainer.get_gate_analysis()
                if gate_analysis and gate_analysis['final_factorized'] is not None:
                    print(f"Gate Balance - Factorized: {gate_analysis['final_factorized']:.2%}, "
                          f"Patch: {gate_analysis['final_patch']:.2%}")
            
            global_epoch_offset = sum([len(self.metrics_history['stages'][i]['train_loss']) 
                                      for i in range(stage_idx)]) if stage_idx > 0 else 0
            
            if (epoch + 1) % args.validation_interval == 0:
                val_metrics = self._validate_epoch(
                    trainer, val_loader, embedding_cache, device, stage_idx
                )
                
                val_loss = val_metrics['val_loss']
                stage_metrics['val_loss'].append(val_loss)
                stage_metrics['val_occupancy'].append(val_metrics.get('val_occupancy', 0))
                stage_metrics['val_density'].append(val_metrics.get('val_density', 0))

                if rank == 0:
                    print(f"Validation Loss: {val_loss:.4f}")
                    print(f"Occupancy: {val_metrics.get('val_occupancy', 0):.2%}")
                    print(f"Density: {val_metrics.get('val_density', 0):.4f}")
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        print(f"✓ New best validation loss: {best_val_loss:.4f}")
                    
                    if torch.cuda.is_available():
                        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                        print(f"Peak Memory: {peak_memory:.2f} GB")

                if rank == 0:
                    print_main("\nGenerating validation visualizations...")
                    
                    sample_descriptions = [
                        "A wooden chair with four legs",
                        "A simple ceramic mug",
                        "A spherical ball",
                        "A rectangular table"
                    ][:4]

                    t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
                    t5_model = T5EncoderModel.from_pretrained('t5-base').to(device)
                    
                    try:
                        self.visualizer.visualize_epoch_samples(
                            trainer=trainer,
                            epoch=epoch + 1,
                            descriptions=sample_descriptions,
                            embedding_cache=embedding_cache,
                            context_encoder=t5_model,
                            tokenizer=t5_tokenizer,
                            granularity=granularity,
                            device=device,
                            num_samples=len(sample_descriptions),
                            view_angles=[(30, 45), (60, 120), (0, 0)],
                            show_slices=True,
                            show_progress=False
                        )
                        print_main("✓ Visualization saved successfully")
                    except Exception as e:
                        print_main(f"⚠ Visualization failed: {e}")
                    finally:
                        del t5_model
                        del t5_tokenizer
                        torch.cuda.empty_cache()
                
                if is_distributed:
                    dist.barrier()
            
            if rank == 0:
                self.plot_loss_curves(stage_idx, stage_metrics, global_epoch_offset, epoch + 1)
            
            if rank == 0:
                print("="*60)
        
        stage_metrics['best_val_loss'] = best_val_loss
        self.metrics_history['stages'][stage_idx] = stage_metrics
        
        self.save_stage_checkpoint(model, stage_idx, device)
        
        del model
        del trainer
        torch.cuda.empty_cache()
        
        if is_distributed:
            dist.barrier()
        
        return stage_metrics

    def _train_epoch(self, trainer, dataloader, embedding_cache, device, epoch, stage_idx):
        """Training epoch."""
        trainer.model.train()
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(dataloader):
            x0, descriptions, file_stems = batch_data
            
            x0 = x0.to(device, dtype=MASTER_DTYPE)
            context = embedding_cache.get_batch_embeddings(descriptions, device)
            
            low_res_batch = None
            if stage_idx > 0 and trainer.low_res_contexts:
                low_res_batch = []
                for stem in file_stems:
                    if stem in trainer.low_res_contexts:
                        low_res_batch.append(
                            trainer.low_res_contexts[stem].to(device, dtype=MASTER_DTYPE).detach()
                        )
                    else:
                        print_main(f"⚠ Warning: Missing low-res context for {stem}")
                        low_res_batch.append(
                            torch.zeros(1, x0.shape[2], x0.shape[3], x0.shape[4],
                                       device=device, dtype=MASTER_DTYPE)
                        )
                
                if low_res_batch:
                    low_res_batch = torch.stack(low_res_batch)
            
            loss = trainer.train_step(x0, context, low_res_batch)
            epoch_loss += loss
            num_batches += 1
            
            if rank == 0 and (batch_idx + 1) % 50 == 0:
                avg_loss = epoch_loss / num_batches
                print(f"  Batch {batch_idx + 1}/{len(dataloader)} - Avg Loss: {avg_loss:.4f}")
        
        if is_distributed:
            metrics = torch.tensor([epoch_loss, num_batches], device=device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            epoch_loss = metrics[0].item()
            num_batches = int(metrics[1].item())
        
        avg_loss = epoch_loss / num_batches
        return {
            'train_loss': avg_loss,
            'num_batches': num_batches
        }

    def _validate_epoch(self, trainer, dataloader, embedding_cache, device, stage_idx):
        """Validation epoch."""
        trainer.model.eval()
        val_loss = 0
        total_occupancy = 0
        total_density = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                if num_batches >= 10:
                    break
                
                x0, descriptions, file_stems = batch_data
                
                x0 = x0.to(device, dtype=MASTER_DTYPE)
                context = embedding_cache.get_batch_embeddings(descriptions, device)
                
                low_res_batch = None
                if stage_idx > 0 and trainer.low_res_contexts:
                    low_res_batch = []
                    for stem in file_stems:
                        if stem in trainer.low_res_contexts:
                            low_res_batch.append(
                                trainer.low_res_contexts[stem].to(device, dtype=MASTER_DTYPE).detach()
                            )
                    
                    if low_res_batch:
                        low_res_batch = torch.stack(low_res_batch)
                
                batch_loss = trainer.compute_loss(x0, context, low_res_batch, backward=False)
                val_loss += batch_loss
                
                threshold = 0.5
                occupancy = (x0 > threshold).float().mean().item()
                density = x0.mean().item()
                total_occupancy += occupancy
                total_density += density
                
                num_batches += 1
        
        if is_distributed:
            metrics = torch.tensor([val_loss, total_occupancy, total_density, num_batches], 
                                  device=device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            val_loss = metrics[0].item()
            total_occupancy = metrics[1].item()
            total_density = metrics[2].item()
            num_batches = int(metrics[3].item())
        
        return {
            'val_loss': val_loss / num_batches if num_batches > 0 else float('inf'),
            'val_occupancy': total_occupancy / num_batches if num_batches > 0 else 0,
            'val_density': total_density / num_batches if num_batches > 0 else 0
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print_main("\n" + "="*60)
    print_main("Layer-by-Layer 3D Diffusion - V5 with Hybrid Causal Attention")
    print_main("="*60)
    
    # Create initial dataset for T5 caching
    train_dataset_3d = VoxelDataset(
        npy_folder_path=f'/home/ad.msoe.edu/benzshawelt/Kedziora/kedziora_research/description_data_generation/objaverse_parallel/objaverse_voxels/{granularity}', 
        description_folder_path='/home/ad.msoe.edu/benzshawelt/Kedziora/kedziora_research/description_data_generation/objaverse_parallel/objaverse_descriptions', 
        granularity=granularity, 
        test_count=0,
        enable_fuzzy_matching=args.enable_fuzzy_matching
    )

    val_dataset_3d = VoxelDataset(
        npy_folder_path=f'/home/ad.msoe.edu/benzshawelt/Kedziora/kedziora_research/description_data_generation/objaverse_parallel/objaverse_voxels/{granularity}', 
        description_folder_path='/home/ad.msoe.edu/benzshawelt/Kedziora/kedziora_research/description_data_generation/objaverse_parallel/objaverse_descriptions', 
        granularity=granularity, 
        test_count=400,
        enable_fuzzy_matching=args.enable_fuzzy_matching
    )

    # Initialize T5
    t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
    t5_model = T5EncoderModel.from_pretrained('t5-base').to(device)
    t5_model.eval()
    print_main(f"T5 model loaded with {sum(p.numel() for p in t5_model.parameters()):,} parameters")

    # Setup T5 Embedding Cache
    print_main("\n" + "="*60)
    print_main("Setting up T5 Embedding Cache")
    print_main("="*60)
    embedding_cache = T5EmbeddingCache(cache_dir='./t5_cache', max_length=77)
    embedding_cache.load_cache()

    if not args.skip_cache_scan:
        print_main("Precomputing embeddings for training dataset...")
        embedding_cache.precompute_embeddings(
            texts=train_dataset_3d,
            tokenizer=t5_tokenizer,
            model=t5_model,
            device=device,
            batch_size=256
        )

        print_main("Precomputing embeddings for validation dataset...")
        embedding_cache.precompute_embeddings(
            texts=val_dataset_3d,
            tokenizer=t5_tokenizer,
            model=t5_model,
            device=device,
            batch_size=256
        )
    else:
        print_main("Skipping cache scan (--skip_cache_scan flag set)")

    train_dataset_3d.save_miss_report(output_path='train_description_miss_report.txt')
    val_dataset_3d.save_miss_report(output_path='val_description_miss_report.txt')

    t5_model = t5_model.cpu()
    embedding_cache.set_embedding_model(t5_model, t5_tokenizer)
    print_main("✓ T5 embeddings cached!\n")

    # Progressive Training
    print_main("\n" + "="*60)
    print_main("Starting Progressive Training (V5 - Hybrid Attention)")
    print_main("="*60)

    granularities = args.granularities
    epochs_per_stage = args.epochs_per_stage

    progressive_trainer = FSDPProgressiveGranularityTrainer(
        granularities=granularities
    )

    results = progressive_trainer.train_all_stages(
        epochs_per_stage=epochs_per_stage,
        device=device,
        embedding_cache=embedding_cache
    )

    metrics_history = results['metrics_history']
    checkpoints = results['checkpoints']

    print_main("\n" + "="*60)
    print_main("Training Complete!")
    print_main("="*60)

    # Save metrics
    if rank == 0:
        with open('progressive_training_metrics_v5.json', 'w') as f:
            metrics_to_save = {
                'stages': {},
                'global_train_loss': metrics_history['global_train_loss'],
                'global_val_loss': metrics_history['global_val_loss'],
                'global_epochs': metrics_history['global_epochs'],
                'model_version': 'v5_hybrid_attention'
            }
            
            for stage_idx, stage_data in metrics_history['stages'].items():
                metrics_to_save['stages'][str(stage_idx)] = {
                    'granularity': stage_data['granularity'],
                    'train_loss': stage_data['train_loss'],
                    'val_loss': stage_data['val_loss'],
                    'val_occupancy': stage_data['val_occupancy'],
                    'val_density': stage_data['val_density'],
                    'best_val_loss': stage_data['best_val_loss'],
                    'epochs': stage_data['epochs']
                }
            
            json.dump(metrics_to_save, f, indent=2)
        print("✓ Training metrics saved to: progressive_training_metrics_v5.json")

    # Plot final training curves
    if rank == 0:
        print("\nGenerating final training curves...")
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(metrics_history['global_train_loss'], alpha=0.7, linewidth=1)
        
        epoch_boundaries = [0]
        for stage_idx in range(len(progressive_trainer.granularities)):
            epochs = epochs_per_stage[stage_idx]
            epoch_boundaries.append(epoch_boundaries[-1] + epochs)
        
        for i, boundary in enumerate(epoch_boundaries[1:-1], 1):
            ax1.axvline(boundary, color='red', linestyle='--', alpha=0.5, linewidth=2)
            mid_point = (epoch_boundaries[i-1] + epoch_boundaries[i]) / 2
            gran = progressive_trainer.granularities[i-1]
            ax1.text(mid_point, ax1.get_ylim()[1] * 0.95, f'{gran}³', 
                    ha='center', fontsize=12, fontweight='bold')
        
        mid_point = (epoch_boundaries[-2] + epoch_boundaries[-1]) / 2
        gran = progressive_trainer.granularities[-1]
        ax1.text(mid_point, ax1.get_ylim()[1] * 0.95, f'{gran}³', 
                ha='center', fontsize=12, fontweight='bold')
        
        ax1.set_xlabel('Global Epoch', fontsize=12)
        ax1.set_ylabel('Training Loss', fontsize=12)
        ax1.set_title('Progressive Training Loss - V5 Hybrid Attention', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        colors = ['blue', 'green', 'orange', 'purple']
        
        ax2 = fig.add_subplot(gs[1, 0])
        for stage_idx, stage_data in metrics_history['stages'].items():
            gran = stage_data['granularity']
            epochs = stage_data['epochs']
            losses = stage_data['train_loss']
            ax2.plot(epochs, losses, label=f'{gran}³', alpha=0.8, 
                    color=colors[stage_idx % len(colors)], linewidth=2)
        ax2.set_xlabel('Stage Epoch', fontsize=10)
        ax2.set_ylabel('Training Loss', fontsize=10)
        ax2.set_title('Per-Stage Training Loss', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs[1, 1])
        for stage_idx, stage_data in metrics_history['stages'].items():
            gran = stage_data['granularity']
            val_epochs = list(range(args.validation_interval, 
                                   len(stage_data['train_loss']) + 1, 
                                   args.validation_interval))
            val_losses = stage_data['val_loss']
            ax3.plot(val_epochs, val_losses, marker='o', label=f'{gran}³', 
                    alpha=0.8, color=colors[stage_idx % len(colors)], linewidth=2)
        ax3.set_xlabel('Stage Epoch', fontsize=10)
        ax3.set_ylabel('Validation Loss', fontsize=10)
        ax3.set_title('Per-Stage Validation Loss', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4 = fig.add_subplot(gs[1, 2])
        stage_names = [f"{metrics_history['stages'][i]['granularity']}³" 
                       for i in range(len(progressive_trainer.granularities))]
        best_losses = [metrics_history['stages'][i]['best_val_loss'] 
                       for i in range(len(progressive_trainer.granularities))]
        bars = ax4.bar(stage_names, best_losses, color=colors[:len(stage_names)], alpha=0.7)
        ax4.set_ylabel('Best Validation Loss', fontsize=10)
        ax4.set_title('Best Val Loss per Stage', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        for bar, loss in zip(bars, best_losses):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{loss:.4f}', ha='center', va='bottom', fontsize=9)
        
        ax5 = fig.add_subplot(gs[2, 0])
        for stage_idx, stage_data in metrics_history['stages'].items():
            gran = stage_data['granularity']
            val_epochs = list(range(args.validation_interval, 
                                   len(stage_data['train_loss']) + 1, 
                                   args.validation_interval))
            occupancy = stage_data['val_occupancy']
            if occupancy:
                ax5.plot(val_epochs, occupancy, marker='s', label=f'{gran}³', 
                        alpha=0.8, color=colors[stage_idx % len(colors)], linewidth=2)
        ax5.set_xlabel('Stage Epoch', fontsize=10)
        ax5.set_ylabel('Occupancy Rate', fontsize=10)
        ax5.set_title('Validation Voxel Occupancy', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        ax6 = fig.add_subplot(gs[2, 1])
        for stage_idx, stage_data in metrics_history['stages'].items():
            gran = stage_data['granularity']
            val_epochs = list(range(args.validation_interval, 
                                   len(stage_data['train_loss']) + 1, 
                                   args.validation_interval))
            density = stage_data['val_density']
            if density:
                ax6.plot(val_epochs, density, marker='^', label=f'{gran}³', 
                        alpha=0.8, color=colors[stage_idx % len(colors)], linewidth=2)
        ax6.set_xlabel('Stage Epoch', fontsize=10)
        ax6.set_ylabel('Mean Density', fontsize=10)
        ax6.set_title('Validation Voxel Density', fontsize=12, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        summary_text = "Training Summary (V5)\n" + "="*30 + "\n\n"
        total_epochs = sum(epochs_per_stage)
        summary_text += f"Model: V5 Hybrid Attention\n"
        summary_text += f"Total Epochs: {total_epochs}\n"
        summary_text += f"Stages: {len(progressive_trainer.granularities)}\n\n"
        
        for stage_idx, gran in enumerate(progressive_trainer.granularities):
            stage_data = metrics_history['stages'][stage_idx]
            summary_text += f"Stage {stage_idx + 1} ({gran}³):\n"
            summary_text += f"  Final Train Loss: {stage_data['train_loss'][-1]:.4f}\n"
            summary_text += f"  Best Val Loss: {stage_data['best_val_loss']:.4f}\n\n"
        
        ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle('Progressive Granularity Training - V5 Hybrid Attention Metrics', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.savefig('progressive_training_curves_v5.png', dpi=150, bbox_inches='tight')
        print("✓ Training curves saved to: progressive_training_curves_v5.png")
        plt.show()
        plt.close()
        
        print("\n" + "="*60)
        print("All Done!")
        print("="*60)
        print("\nKey files saved:")
        for stage_idx, checkpoint_path in checkpoints.items():
            print(f"  - {checkpoint_path} (stage {stage_idx})")
        print("  - progressive_training_metrics_v5.json")
        print("  - progressive_training_curves_v5.png")
        print("  - ./t5_cache/")
        print("\n" + "="*60)

    # Cleanup
    if is_distributed:
        dist.destroy_process_group()
    torch.cuda.empty_cache()