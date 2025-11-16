# %%
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
from tqdm import tqdm



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
    parser.add_argument('--current_layer_only', action='store_true',
                    help='Use only current layer low-res context (vs all layers)')
    parser.add_argument('--enable_fuzzy_matching', action='store_true',
        help='Enable fuzzy matching for missing descriptions (slower initialization)')

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

class T5EmbeddingCache:
    """
    Unified T5 embedding cache using SQLite backend with LRU memory cache.
    Drop-in replacement for both HDF5 and SQL versions.
    """
    
    def __init__(self, cache_dir='./t5_cache', max_length=77, memory_cache_size=1000):
        """
        Initialize T5 embedding cache with SQLite backend.
        
        Args:
            cache_dir: Directory to store cache database
            max_length: Maximum sequence length for embeddings
            memory_cache_size: Number of embeddings to keep in RAM (LRU)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_length = max_length
        self.db_path = self.cache_dir / 'embeddings.db'
        self.memory_cache_size = memory_cache_size
        
        # Small LRU cache for frequently accessed embeddings
        self._memory_cache = {}
        self._cache_order = []
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database with optimizations"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Create table with index on text_hash for fast lookups
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                text_hash TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create index for faster lookups (if not exists)
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_text_hash 
            ON embeddings(text_hash)
        ''')
        
        # Optimize SQLite for our use case
        cursor.execute('PRAGMA journal_mode=WAL')  # Better concurrent access
        cursor.execute('PRAGMA synchronous=NORMAL')  # Faster writes
        cursor.execute('PRAGMA cache_size=-64000')  # 64MB cache
        
        conn.commit()
        conn.close()
    
    def _get_cache_key(self, text):
        """Generate a unique key for a text description"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _add_to_memory_cache(self, key, embedding):
        """Add to memory cache with LRU eviction"""
        if key in self._memory_cache:
            # Move to end (most recent)
            self._cache_order.remove(key)
        
        self._memory_cache[key] = embedding
        self._cache_order.append(key)
        
        # Evict oldest if cache is full
        if len(self._memory_cache) > self.memory_cache_size:
            oldest_key = self._cache_order.pop(0)
            del self._memory_cache[oldest_key]
    
    def load_cache(self):
        """
        Load cache from disk (prints statistics).
        For API compatibility - SQLite is always persistent.
        """
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
        """
        Save cache to disk (prints statistics).
        For API compatibility - SQLite writes are immediate.
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM embeddings')
        count = cursor.fetchone()[0]
        conn.close()
        print(f"Database contains {count:,} cached embeddings")
    
    def get_embedding(self, text):
        """
        Get embedding from cache (memory -> disk).
        
        Args:
            text: Text string to get embedding for
            
        Returns:
            torch.Tensor or None: Cached embedding [max_length, 768] or None if not found
        """
        key = self._get_cache_key(text)
        
        # Check memory cache first (fast path)
        if key in self._memory_cache:
            return self._memory_cache[key]
        
        # Check disk cache (slower path)
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute('SELECT embedding FROM embeddings WHERE text_hash = ?', (key,))
        result = cursor.fetchone()
        conn.close()
        
        if result is not None:
            # Deserialize from database
            embedding = torch.from_numpy(
                np.frombuffer(result[0], dtype=np.float16).reshape(self.max_length, 768)
            )
            
            # Add to memory cache for future access
            self._add_to_memory_cache(key, embedding)
            return embedding
        
        return None
    
    def add_embedding(self, text, embedding):
        """
        Add embedding to cache (both disk and memory).
        
        Args:
            text: Text string
            embedding: torch.Tensor [max_length, 768]
        """
        key = self._get_cache_key(text)
        
        # Serialize to bytes (float16 for space efficiency)
        embedding_bytes = embedding.cpu().numpy().astype(np.float16).tobytes()
        
        # Write to disk
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
    
    def precompute_embeddings_from_dataset(self, dataset, tokenizer, model, device, batch_size=32):
        """
        Precompute embeddings for all descriptions in a VoxelDataset.
        Loads descriptions lazily to save memory.
        
        Args:
            dataset: VoxelDataset instance (or any dataset that returns (data, description))
            tokenizer: T5 tokenizer
            model: T5 encoder model
            device: Device to run on
            batch_size: Batch size for embedding computation
        """
        print(f"Scanning {len(dataset)} samples for uncached descriptions...")
        
        # First pass: collect unique uncached descriptions
        uncached_descriptions = set()
        
        for idx in tqdm(range(len(dataset)), desc="Scanning for uncached descriptions"):
            # Load just the description (lazy loading)
            _, description = dataset[idx]
            
            # Check if already cached
            if self.get_embedding(description) is None:
                uncached_descriptions.add(description)
        
        if not uncached_descriptions:
            print("All embeddings already cached!")
            return
        
        print(f"Found {len(uncached_descriptions)} unique uncached descriptions")
        print(f"Computing embeddings...")
        
        # Convert to list for batching
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
                
                # Add each embedding to cache
                for text, emb in zip(batch_texts, embeddings):
                    self.add_embedding(text, emb)
                
                # Cleanup GPU memory
                del text_inputs, embeddings
                if torch.cuda.is_available() and (i // batch_size) % 10 == 0:
                    torch.cuda.empty_cache()
        
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"✓ Cached {len(uncached_list)} embeddings to {self.db_path}")
    
    def precompute_embeddings(self, texts, tokenizer, model, device, batch_size=128):
        """
        Precompute embeddings for a list of texts OR a Dataset object.
        
        Args:
            texts: List of text strings OR a Dataset object (VoxelDataset)
            tokenizer: T5 tokenizer
            model: T5 encoder model
            device: Device to run on
            batch_size: Batch size for embedding computation
        """
        # Check if texts is a Dataset
        if hasattr(texts, '__getitem__') and hasattr(texts, '__len__') and not isinstance(texts, (list, tuple, set)):
            # It's a dataset-like object
            return self.precompute_embeddings_from_dataset(texts, tokenizer, model, device, batch_size)
        
        # Original behavior for list of texts
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
                
                # Add each embedding to cache
                for text, emb in zip(batch_texts, embeddings):
                    self.add_embedding(text, emb)
                
                # Cleanup GPU memory
                del text_inputs, embeddings
                if torch.cuda.is_available() and (i // batch_size) % 10 == 0:
                    torch.cuda.empty_cache()
        
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"✓ Cached {len(uncached_texts)} embeddings to {self.db_path}")
    
    def get_batch_embeddings(self, texts, device):
        """
        Get batch of embeddings from cache.
        
        Args:
            texts: List of text strings
            device: Device to move embeddings to
            
        Returns:
            torch.Tensor: Stacked embeddings [batch_size, max_length, 768]
        """
        embeddings = []
        for text in texts:
            emb = self.get_embedding(text)
            if emb is None:
                raise ValueError(f"Embedding not found in cache for text: {text[:50]}...")
            embeddings.append(emb)
        
        # Determine dtype based on environment
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float16
        
        return torch.stack(embeddings).to(device, dtype=dtype)
    
    def clear_memory_cache(self):
        """Clear the in-memory LRU cache (keeps disk cache intact)"""
        self._memory_cache.clear()
        self._cache_order.clear()
        print("Memory cache cleared")
    
    def get_stats(self):
        """Get cache statistics"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM embeddings')
        total_cached = cursor.fetchone()[0]
        
        cursor.execute('SELECT pg_size FROM pragma_page_size()')
        page_size = cursor.fetchone()[0]
        
        cursor.execute('SELECT pg_count FROM pragma_page_count()')
        page_count = cursor.fetchone()[0]
        
        conn.close()
        
        db_size_mb = (page_size * page_count) / (1024 * 1024)
        memory_cached = len(self._memory_cache)
        
        return {
            'total_embeddings': total_cached,
            'memory_cached': memory_cached,
            'database_size_mb': db_size_mb,
            'memory_cache_size': self.memory_cache_size,
            'database_path': str(self.db_path)
        }
    
    def print_stats(self):
        """Print cache statistics"""
        stats = self.get_stats()
        print("\n" + "="*60)
        print("T5 Embedding Cache Statistics")
        print("="*60)
        print(f"Total embeddings in database: {stats['total_embeddings']:,}")
        print(f"Embeddings in memory cache:   {stats['memory_cached']:,}")
        print(f"Database size:                {stats['database_size_mb']:.2f} MB")
        print(f"Memory cache capacity:        {stats['memory_cache_size']:,}")
        print(f"Database path:                {stats['database_path']}")
        print("="*60 + "\n")

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
        # Clone before chunking to avoid CUDA Graphs conflict
        scale_shift_out = self.scale_shift(cond).clone()
        scale, shift = scale_shift_out.chunk(2, dim=-1)
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
                 attention_resolutions=[8, 16], use_low_res_context=False, current_low_res_layer_context=False, context_encoder_mode='spatial_aware'):
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
        self.use_low_res_context = use_low_res_context
        self.current_low_res_layer_context = current_low_res_layer_context
        self.context_encoder_mode = context_encoder_mode

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

        if use_low_res_context:
            self.low_res_encoder = LowResContextEncoder(
                in_channels=1,
                base_channels=64,
                mode=self.context_encoder_mode,
            )
            
            # Additional cross-attention for low-res context
            self.low_res_cross_attn = MultiHeadCrossAttention(
                channels=self.model_channels,
                context_dim=128,  # From low_res_encoder
                num_heads=4
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

    def forward(self, x, t, l, context, prev_layers=None, low_res_context=None):
        """
        Forward pass with cross-attention to all previous layers.
        
        Args:
            x: [B, C, H, W] - noisy input for current layer
            t: [B] - diffusion timestep
            l: [B] - layer index (0 to granularity-1)
            context: [B, text_context_dim] - text embeddings
            prev_layers: [B, N, C, H, W] or None - ALL previous clean layers
            low_res_context: [B, 1, H_low, W_low] or None - low resolution context image
        
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


        if self.use_low_res_context and low_res_context is not None:
            # Encode low-res context into per-layer features
            low_res_features = self.low_res_encoder(
                low_res_context, 
                self.granularity
            )  # [B, granularity, 128]
            
            if self.current_low_res_layer_context: 
                # Get features for current layer
                batch_size = x.shape[0]
                layer_indices = l.long()
                current_low_res_feat = low_res_features[
                    torch.arange(batch_size), 
                    layer_indices
                ]  # [B, 128]
                current_low_res_feat = current_low_res_feat.unsqueeze(1)  # [B, 1, 128]
                
                # Apply cross-attention with low-res context
                h = self.low_res_cross_attn(h, current_low_res_feat)
            else:
                # Apply cross-attention with all low-res context features
                h = self.low_res_cross_attn(h, low_res_features)

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
        
        # Track description misses for debugging
        self.description_misses = []  
        
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
        
        # Only cache description filenames if fuzzy matching is enabled
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
        print(f"Descriptions will be loaded on-demand from {description_folder_path}")
    
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
            # Only do fuzzy matching if enabled
            if self.enable_fuzzy_matching and self.available_descriptions is not None:
                closest_matches = get_close_matches(filename_stem, self.available_descriptions, n=1, cutoff=0.6)
                closest_match = closest_matches[0] if closest_matches else "No close match found"
                
                miss_info = {
                    'missing': filename_stem,
                    'closest_match': closest_match
                }
                self.description_misses.append(miss_info)

                # if a close match was found, try to load it
                
                if closest_matches:
                    print(f"Fuzzy matched missing description: {filename_stem} -> {closest_match}")
                    matched_file = self.description_folder_path / f"{closest_match}.txt"
                    try:
                        with open(matched_file, 'r') as f:
                            return f.read().strip()
                    except Exception as e:
                        print(f"Error loading matched file {matched_file}: {e}")
                else:
                    print(f"No close match found for missing description: {filename_stem}")
            else:
                print(f"Description file not found: {desc_file}")

                # Simple miss tracking without fuzzy matching
                self.description_misses.append({
                    'missing': filename_stem,
                    'closest_match': 'N/A (fuzzy matching disabled)'
                })
            
            
            # randomly select a default description
            return random.choice(self.default_description)

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # Load voxel data (already on-demand)
        voxel_grid = np.load(self.file_list[idx]).astype(np.float16)
        assert voxel_grid.shape == (self.granularity, self.granularity, self.granularity), \
            f"Voxel grid shape {voxel_grid.shape} does not match expected shape {(self.granularity, self.granularity, self.granularity)}"
        
        if self.transform:
            voxel_grid = self.transform(voxel_grid)
        
        voxel_grid = torch.from_numpy(voxel_grid).unsqueeze(0)  # Add channel dimension
        
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

class LayerXLayerDiffusionTrainerV3:
    """
    Unified trainer supporting both regular and FSDP training,
    with optional low-resolution context conditioning.
    Consolidates LayerXLayerDiffusionTrainerV2 and FSDPCompatibleTrainer.
    """
    def __init__(self, model: LayerXLayerDiffusionModelV2, diffusion, scheduler, 
                 layer_by_layer_convergence=True, teacher_forcing=True, 
                 use_ddim=True, ddim_steps=50, ddim_eta=0.0, 
                 low_res_contexts=None):
        """
        Args:
            model: LayerXLayerDiffusionModelV2 instance
            diffusion: ForwardDiffusion instance
            scheduler: Learning rate scheduler
            layer_by_layer_convergence: Use layer convergence during generation
            teacher_forcing: Use ground truth layers vs predicted layers
            use_ddim: Use DDIM sampling (faster) vs full diffusion
            ddim_steps: Number of steps for DDIM sampling
            ddim_eta: Stochasticity parameter for DDIM (0.0 = deterministic)
            low_res_contexts: Dict of precomputed low-res contexts (for progressive training)
        """
        self.model = model
        self.diffusion = diffusion
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        self.scheduler = scheduler
        self.layer_by_layer_convergence = layer_by_layer_convergence
        self.teacher_forcing = teacher_forcing
        self.use_ddim = use_ddim
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        self.low_res_contexts = low_res_contexts or {}

    def compute_loss(self, x0, context, low_res_context=None, backward=True):
        """
        Compute loss with optional backward pass.
        
        Args:
            x0: [B, C, H, W, D] - full 3D object
            context: [B, seq_len, context_dim] - text embeddings
            low_res_context: Optional [B, C, H, W, D] - upsampled low-res conditioning
            backward: Whether to perform backward pass and optimization
        
        Returns:
            avg_loss: Average loss across all layers
        """
        if backward:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()

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
            predicted_noise = self.model(xt, t, l, context, prev_layers, low_res_context)

            # Compute loss
            loss = F.mse_loss(predicted_noise.float(), noise.float())
            total_loss += loss.detach() if backward else loss

            if backward:
                loss.backward()

            # Add current layer to history
            if self.teacher_forcing:
                prev_layers_list.append(current_layer.detach())
            else:
                # Use denoised prediction
                alpha_hat_t = self.diffusion.alpha_hats[t].view(-1, 1, 1, 1)
                denoised = (xt - torch.sqrt(1 - alpha_hat_t) * predicted_noise) / torch.sqrt(alpha_hat_t)
                prev_layers_list.append(denoised.detach())

        # Optimize if backward pass requested
        if backward:
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

        avg_loss = total_loss / self.model.granularity
        return avg_loss.item() if not backward else avg_loss

    def train_step(self, x0, context, low_res_context=None):
        """
        Training step with backward pass.
        
        Args:
            x0: [B, C, H, W, D] - full 3D object
            context: [B, seq_len, context_dim] - text embeddings
            low_res_context: Optional [B, C, H, W, D] - upsampled low-res conditioning
        
        Returns:
            avg_loss: Average loss across all layers
        """
        return self.compute_loss(x0, context, low_res_context, backward=True)

    def validate(self, dataloader, context_cache, device, max_batches=None):
        """
        Validation loop with full cross-attention.
        
        Args:
            dataloader: Validation DataLoader
            context_cache: T5EmbeddingCache instance
            device: Device to run on
            max_batches: Maximum number of batches to validate (None = all)
        
        Returns:
            dict: Validation metrics
        """
        self.model.eval()
        total_loss = 0
        total_occupancy = 0
        total_density = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (x0, descriptions) in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                
                x0 = x0.to(device)
                context = context_cache.get_batch_embeddings(descriptions, device)
                
                # Get low-res context if available
                low_res_batch = None
                if self.low_res_contexts:
                    batch_start = batch_idx * dataloader.batch_size
                    low_res_batch = []
                    for i in range(len(descriptions)):
                        idx = batch_start + i
                        if idx in self.low_res_contexts:
                            low_res_batch.append(
                                self.low_res_contexts[idx].to(device)
                            )
                    
                    if low_res_batch:
                        low_res_batch = torch.stack(low_res_batch)
                
                # Compute loss without backward pass
                loss = self.compute_loss(x0, context, low_res_batch, backward=False)
                total_loss += loss
                
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

class LowResContextEncoder(nn.Module):
    """
    Enhanced encoder with better spatial and inter-layer awareness.
    Provides richer context for conditioning at higher resolutions.
    """
    def __init__(self, in_channels=1, base_channels=64, mode='spatial_aware'):
        super().__init__()
        self.mode = mode  # 'spatial_aware', 'simple', or 'hierarchical'
        
        # Deeper 3D encoder with residual connections
        self.encoder = nn.ModuleList([
            self._make_conv3d_block(in_channels, base_channels),
            self._make_conv3d_block(base_channels, base_channels * 2, stride=2),
            self._make_conv3d_block(base_channels * 2, base_channels * 4, stride=2),
        ])
        
        # Learnable layer positional embeddings
        self.layer_pos_embedding = nn.Embedding(256, 64)  # Support up to 256 layers
        
        if mode == 'spatial_aware':
            # Preserve spatial information via learned aggregation
            self.spatial_aggregator = nn.Sequential(
                nn.Conv2d(base_channels * 4, 256, 1),
                nn.GroupNorm(8, 256),
                nn.SiLU(),
                nn.Conv2d(256, 128, 1)
            )
            # Spatial position attention
            self.spatial_attention = nn.MultiheadAttention(
                embed_dim=128,
                num_heads=4,
                batch_first=True
            )
        elif mode == 'hierarchical':
            # Multi-scale feature fusion
            self.multiscale_fusion = nn.ModuleList([
                nn.Conv2d(base_channels, 32, 1),
                nn.Conv2d(base_channels * 2, 32, 1),
                nn.Conv2d(base_channels * 4, 64, 1),
            ])
            self.fusion_proj = nn.Linear(128, 128)
        else:  # 'simple' mode (improved version of original)
            self.layer_projection = nn.Sequential(
                nn.Linear(base_channels * 4, 256),
                nn.LayerNorm(256),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128)
            )
        
        # Inter-layer transformer for capturing vertical dependencies
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
    
    def _make_conv3d_block(self, in_ch, out_ch, stride=1):
        """Create a residual 3D conv block"""
        block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
        )
        
        # Residual connection
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv3d(in_ch, out_ch, 1, stride=stride)
        else:
            self.shortcut = nn.Identity()
        
        return block
    
    def forward(self, low_res_voxels, target_granularity):
        """
        Args:
            low_res_voxels: [B, C, H, W, D] - upsampled previous resolution
            target_granularity: int - number of layers at current resolution
        Returns:
            layer_features: [B, target_granularity, 128]
        """
        B = low_res_voxels.shape[0]
        
        # Hierarchical feature extraction
        features_pyramid = []
        x = low_res_voxels
        
        for i, block in enumerate(self.encoder):
            if i > 0:  # Apply shortcut for residual
                identity = self.encoder[i-1].shortcut(x) if hasattr(self.encoder[i-1], 'shortcut') else x
                x = block(x) + identity
            else:
                x = block(x)
            features_pyramid.append(x)
        
        features = features_pyramid[-1]  # [B, C', H', W', D']
        
        # Extract per-layer features with smooth interpolation
        layer_features = []
        
        for layer_idx in range(target_granularity):
            # Smooth interpolation instead of hard indexing
            z_coord = (layer_idx + 0.5) / target_granularity * features.shape[-1]
            z_idx_low = int(z_coord)
            z_idx_high = min(z_idx_low + 1, features.shape[-1] - 1)
            alpha = z_coord - z_idx_low
            
            # Linearly interpolate between adjacent slices
            layer_feat = (1 - alpha) * features[:, :, :, :, z_idx_low] + \
                        alpha * features[:, :, :, :, z_idx_high]  # [B, C', H', W']
            
            if self.mode == 'spatial_aware':
                # Preserve spatial structure
                layer_feat = self.spatial_aggregator(layer_feat)  # [B, 128, H', W']
                
                # Flatten spatial dimensions for attention
                B, C, H, W = layer_feat.shape
                layer_feat_flat = layer_feat.flatten(2).transpose(1, 2)  # [B, H*W, 128]
                
                # Self-attention over spatial positions
                layer_feat_attn, _ = self.spatial_attention(
                    layer_feat_flat, layer_feat_flat, layer_feat_flat
                )  # [B, H*W, 128]
                
                # Aggregate with learned attention weights
                layer_feat = layer_feat_attn.mean(dim=1)  # [B, 128]
                
            elif self.mode == 'hierarchical':
                # Multi-scale fusion
                scale_features = []
                for j, (feat, proj) in enumerate(zip(features_pyramid, self.multiscale_fusion)):
                    # Get corresponding slice from each scale
                    z_idx = int(layer_idx * feat.shape[-1] / target_granularity)
                    z_idx = min(z_idx, feat.shape[-1] - 1)
                    
                    feat_slice = feat[:, :, :, :, z_idx]  # [B, C_j, H_j, W_j]
                    feat_proj = proj(feat_slice)  # [B, out_ch, H_j, W_j]
                    feat_pool = F.adaptive_avg_pool2d(feat_proj, (1, 1)).flatten(1)
                    scale_features.append(feat_pool)
                
                layer_feat = torch.cat(scale_features, dim=1)  # [B, 128]
                layer_feat = self.fusion_proj(layer_feat)
                
            else:  # 'simple' mode
                # Improved pooling with more context
                layer_feat = F.adaptive_avg_pool2d(layer_feat, (2, 2))  # [B, C', 2, 2]
                layer_feat = layer_feat.flatten(1)  # [B, C'*4]
                layer_feat = self.layer_projection(layer_feat)  # [B, 128]
            
            layer_features.append(layer_feat)
        
        # Stack all layer features
        layer_features = torch.stack(layer_features, dim=1)  # [B, target_granularity, 128]
        
        # Add learnable positional embeddings
        layer_indices = torch.arange(target_granularity, device=layer_features.device)
        pos_emb = self.layer_pos_embedding(layer_indices).unsqueeze(0)  # [1, target_granularity, 64]
        
        # Concatenate and project back
        layer_features = torch.cat([layer_features, pos_emb.expand(B, -1, -1)], dim=-1)
        layer_features = nn.Linear(128 + 64, 128).to(layer_features.device)(layer_features)
        
        # Apply inter-layer transformer for vertical coherence
        layer_features = self.inter_layer_transformer(layer_features)  # [B, target_granularity, 128]
        
        return layer_features

class FSDPProgressiveGranularityTrainer:
    """FSDP-compatible progressive training manager"""
    
    def __init__(self, granularities=[16, 32, 64], base_config=None):
        self.granularities = granularities
        self.base_config = base_config or {}
        self.stage_checkpoints = {}
        self.low_res_cache = {}

        self.visualizer = TrainingVoxelVisualizer(
            save_dir='./training_visualizations'
        )


        self.metrics_history = {
            'stages': {},  # Per-stage metrics
            'global_train_loss': [],  # All training losses concatenated
            'global_val_loss': [],  # All validation losses
            'global_epochs': [],  # Epoch numbers for global tracking
        }
        
    def train_all_stages(self, epochs_per_stage, device, embedding_cache):
        """Train all granularity stages sequentially and return metrics"""
        print_main("\n" + "="*60)
        print_main("PROGRESSIVE GRANULARITY TRAINING")
        print_main("="*60)
        print_main(f"Stages: {self.granularities}")
        print_main(f"Epochs per stage: {epochs_per_stage}")
        print_main("="*60 + "\n")
        
        for stage_idx in range(len(self.granularities)):
            granularity = self.granularities[stage_idx]
            epochs = epochs_per_stage[stage_idx]
            
            # Create datasets for this granularity
            print_main(f"\nCreating datasets for {granularity}³ resolution...")
            stage_train_dataset = VoxelDataset(
                npy_folder_path=f'/home/ad.msoe.edu/benzshawelt/Kedziora/kedziora_research/description_data_generation/objaverse_parallel/objaverse_voxels/{granularity}',
                description_folder_path='/home/ad.msoe.edu/benzshawelt/Kedziora/kedziora_research/description_data_generation/objaverse_parallel/objaverse_descriptions',
                granularity=granularity,
                test_count=100,  # Adjust as needed
                enable_fuzzy_matching=args.enable_fuzzy_matching
            )
            
            stage_val_dataset = VoxelDataset(
                npy_folder_path=f'/home/ad.msoe.edu/benzshawelt/Kedziora/kedziora_research/description_data_generation/objaverse_parallel/objaverse_voxels/{granularity}',
                description_folder_path='/home/ad.msoe.edu/benzshawelt/Kedziora/kedziora_research/description_data_generation/objaverse_parallel/objaverse_descriptions',
                granularity=granularity,
                test_count=100,
                enable_fuzzy_matching=args.enable_fuzzy_matching
            )
            
            # Train this stage
            stage_metrics = self.train_stage(
                stage_idx=stage_idx,
                train_dataset=stage_train_dataset,
                val_dataset=stage_val_dataset,
                epochs=epochs,
                device=device,
                embedding_cache=embedding_cache
            )
            
            # Add to global metrics
            global_epoch_offset = sum(epochs_per_stage[:stage_idx])
            for epoch_loss in stage_metrics['train_loss']:
                self.metrics_history['global_train_loss'].append(epoch_loss)
            
            # Val losses are sparse (only every validation_interval)
            for val_loss in stage_metrics['val_loss']:
                self.metrics_history['global_val_loss'].append(val_loss)
            
            # Track global epoch numbers for plotting
            stage_epoch_nums = [global_epoch_offset + e for e in stage_metrics['epochs']]
            self.metrics_history['global_epochs'].extend(stage_epoch_nums)
        
        print_main("\n" + "="*60)
        print_main("PROGRESSIVE TRAINING COMPLETE!")
        print_main("="*60)
        print_main("\nTrained models:")
        for stage_idx, granularity in enumerate(self.granularities):
            if stage_idx in self.stage_checkpoints:
                best_loss = self.metrics_history['stages'][stage_idx]['best_val_loss']
                print_main(f"  Stage {stage_idx} ({granularity}³): {self.stage_checkpoints[stage_idx]}")
                print_main(f"    Best Val Loss: {best_loss:.4f}")
        print_main("="*60 + "\n")
        
        # Return complete metrics
        return {
            'metrics_history': self.metrics_history,
            'checkpoints': self.stage_checkpoints,
            'granularities': self.granularities
        }

    def save_stage_checkpoint(self, model, stage_idx, device):
        """Properly save FSDP model state for future use"""
        granularity = self.granularities[stage_idx]
        
        if args.shard_model:
            # Gather full state dict on rank 0
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
                    checkpoint_path = f'stage{stage_idx}_gran{granularity}.pth'
                    torch.save({
                        'model_state_dict': state_dict,
                        'granularity': granularity,
                        'stage_idx': stage_idx
                    }, checkpoint_path)
                    self.stage_checkpoints[stage_idx] = checkpoint_path
                    print(f"✓ Stage {stage_idx} checkpoint saved: {checkpoint_path}")
        else:
            # Non-FSDP case
            checkpoint_path = f'stage{stage_idx}_gran{granularity}.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'granularity': granularity,
                'stage_idx': stage_idx
            }, checkpoint_path)
            self.stage_checkpoints[stage_idx] = checkpoint_path
        
        # Synchronize all ranks
        if is_distributed:
            dist.barrier()
            # Broadcast checkpoint path to all ranks
            if rank == 0:
                path_list = [self.stage_checkpoints[stage_idx]]
            else:
                path_list = [None]
            dist.broadcast_object_list(path_list, src=0)
            self.stage_checkpoints[stage_idx] = path_list[0]
    
    def load_prev_stage_model_cpu(self, stage_idx):
        """Load previous stage model on CPU for inference"""
        if stage_idx == 0 or stage_idx - 1 not in self.stage_checkpoints:
            return None
        
        checkpoint_path = self.stage_checkpoints[stage_idx - 1]
        prev_granularity = self.granularities[stage_idx - 1]
        
        # Load model on CPU (only rank 0 needs to do this)
        if rank == 0:
            # Create model WITHOUT FSDP wrapping
            prev_model = LayerXLayerDiffusionModelV2(  # Use base model for stage 0
                layer_context_dim=64,
                granularity=prev_granularity,
                text_context_dim=768,
                max_context_layers=16,
                in_channels=1,
                model_channels=512,
                context_dim=768,
                attention_resolutions=[8, 16, 32]
            )
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            prev_model.load_state_dict(checkpoint['model_state_dict'])
            prev_model.eval()
            prev_model = prev_model.cpu()  # Keep on CPU
            
            print(f"✓ Loaded stage {stage_idx-1} model on CPU for inference")
            return prev_model
        else:
            return None  # Only rank 0 has the model

    def precompute_low_res_contexts(self, stage_idx, train_dataset, device, 
                                     embedding_cache, batch_size=4):
        """
        Precompute all low-res contexts before training starts.
        Only rank 0 generates, then broadcasts to all ranks.
        """
        if stage_idx == 0:
            return {}
        
        print_main(f"\nPrecomputing low-res contexts for stage {stage_idx}...")
        
        # Check if cache already exists
        cache_path = f'low_res_context_stage{stage_idx}.pth'
        if Path(cache_path).exists():
            print_main(f"✓ Loading existing cache from {cache_path}")
            low_res_contexts = torch.load(cache_path, map_location='cpu')
            print_main(f"✓ Loaded {len(low_res_contexts)} cached contexts")
            return low_res_contexts
        
        # Load previous stage model on CPU (rank 0 only)
        prev_model = self.load_prev_stage_model_cpu(stage_idx)
        
        low_res_contexts = {}
        curr_granularity = self.granularities[stage_idx]
        prev_granularity = self.granularities[stage_idx - 1]
        
        if rank == 0:
            # Create dataset for previous granularity
            prev_dataset = VoxelDataset(
                npy_folder_path=f'/home/ad.msoe.edu/benzshawelt/Kedziora/kedziora_research/description_data_generation/objaverse_parallel/objaverse_voxels/{prev_granularity}',
                description_folder_path='/home/ad.msoe.edu/benzshawelt/Kedziora/kedziora_research/description_data_generation/objaverse_parallel/objaverse_descriptions',
                granularity=prev_granularity,
                test_count=100,
                enable_fuzzy_matching=args.enable_fuzzy_matching
            )
            
            prev_loader = DataLoader(
                prev_dataset, 
                batch_size=batch_size, 
                shuffle=False,
                num_workers=2,
                persistent_workers=True
            )
            
            # Generate low-res voxels using previous stage model
            with torch.no_grad():
                for batch_idx, (x0, descriptions) in enumerate(tqdm(
                    prev_loader, desc="Generating low-res contexts"
                )):
                    # Get embeddings
                    context = embedding_cache.get_batch_embeddings(
                        descriptions, 'cpu'
                    )
                    
                    # Generate using previous model (on CPU)
                    samples = self._fast_sample_cpu(
                        prev_model, context, x0.shape, prev_granularity
                    )
                    
                    # Upsample to current granularity
                    upsampled = F.interpolate(
                        samples,
                        size=(curr_granularity, curr_granularity, curr_granularity),
                        mode='trilinear',
                        align_corners=False
                    )
                    
                    # Store with dataset indices
                    for i, desc in enumerate(descriptions):
                        idx = batch_idx * batch_size + i
                        low_res_contexts[idx] = upsampled[i].cpu()
                    
                    # Save periodically
                    if (batch_idx + 1) % 10 == 0:
                        self._save_context_cache(low_res_contexts, stage_idx)
            
            print(f"✓ Generated {len(low_res_contexts)} low-res contexts")
            
            # Final save
            self._save_context_cache(low_res_contexts, stage_idx)
        
        # Synchronize all ranks
        if is_distributed:
            dist.barrier()
            # All ranks load from disk
            low_res_contexts = self._load_context_cache(stage_idx)
        
        return low_res_contexts
    
    def _fast_sample_cpu(self, model, context, shape, granularity):
        """Fast DDIM sampling on CPU for context generation"""
        model.eval()
        batch_size = shape[0]
        
        # Use DDIM with fewer steps for speed
        ddim_steps = 20
        timestep_interval = 1000 // ddim_steps
        ddim_timesteps = list(range(0, 1000, timestep_interval))
        ddim_timesteps.reverse()
        
        voxel_grid = torch.zeros(
            (batch_size, 1, granularity, granularity, granularity),
            device='cpu'
        )
        
        generated_layers = []
        diffusion = ForwardDiffusion(timesteps=1000, schedule='cosine')
        
        for layer_idx in range(granularity):
            l = torch.full((batch_size,), layer_idx, dtype=torch.float32)
            x = torch.randn((batch_size, 1, granularity, granularity))
            
            if len(generated_layers) > 0:
                prev_layers = torch.stack(generated_layers, dim=1)
            else:
                prev_layers = None
            
            # DDIM sampling
            for i, t in enumerate(ddim_timesteps):
                t_batch = torch.full((batch_size,), t, dtype=torch.float32)
                
                predicted_noise = model(
                    x, t_batch, l, context, prev_layers
                )
                
                alpha_hat_t = diffusion.alpha_hats[t]
                pred_x0 = (x - torch.sqrt(1 - alpha_hat_t) * predicted_noise) / torch.sqrt(alpha_hat_t)
                pred_x0 = pred_x0.clamp(-1, 1)
                
                if i < len(ddim_timesteps) - 1:
                    t_prev = ddim_timesteps[i + 1]
                    alpha_hat_t_prev = diffusion.alpha_hats[t_prev]
                    dir_xt = torch.sqrt(1 - alpha_hat_t_prev) * predicted_noise
                    x = torch.sqrt(alpha_hat_t_prev) * pred_x0 + dir_xt
                else:
                    x = pred_x0
            
            voxel_grid[:, :, :, :, layer_idx] = x
            generated_layers.append(x.detach())
        
        return voxel_grid.clamp(0, 1)
    
    def _save_context_cache(self, contexts, stage_idx):
        """Save context cache to disk"""
        cache_path = f'low_res_context_stage{stage_idx}.pth'
        torch.save(contexts, cache_path)
        print(f"✓ Context cache saved: {cache_path}")
    
    def _load_context_cache(self, stage_idx):
        """Load context cache from disk"""
        cache_path = f'low_res_context_stage{stage_idx}.pth'
        if Path(cache_path).exists():
            return torch.load(cache_path, map_location='cpu')
        return {}
    
    def train_stage(self, stage_idx, train_dataset, val_dataset,
                   epochs, device, embedding_cache):
        """FSDP-compatible stage training"""
        granularity = self.granularities[stage_idx]
        use_low_res = stage_idx > 0
        
        print_main(f"\n{'='*60}")
        print_main(f"STAGE {stage_idx + 1}: Training at {granularity}³")
        if use_low_res:
            prev_gran = self.granularities[stage_idx - 1]
            print_main(f"Using {prev_gran}³ as conditioning")
        print_main(f"{'='*60}\n")
        
        # Precompute low-res contexts if needed
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
            'epochs': list(range(1, epochs + 1))
        }
        
        # Create model (use base model for now, can extend later)
        model = LayerXLayerDiffusionModelV2(
            layer_context_dim=64,
            granularity=granularity,
            text_context_dim=768,
            max_context_layers=16,
            in_channels=1,
            model_channels=512,
            context_dim=768,
            attention_resolutions=[8, 16, 32],
            current_low_res_layer_context=args.current_layer_only
        )
        
        model = initialize_model_weights(model).to(device)

        model = torch.compile(model, mode='default')
        
        # Wrap with FSDP
        if args.shard_model:
            fsdp_config = get_fsdp_config()
            model = FSDP(model, **fsdp_config)
            print_main("✓ Model wrapped with FSDP")
        
        # Create trainer
        diffusion = ForwardDiffusion(timesteps=1000, schedule='cosine')
        trainer = LayerXLayerDiffusionTrainerV3(  # Use unified trainer
            model=model,
            diffusion=diffusion,
            scheduler=None,
            low_res_contexts=low_res_contexts,  # Pass precomputed contexts
            use_ddim=True,
            ddim_steps=50
        )
        
        # Create dataloaders
        batch_size = args.batch_size if args.batch_size is not None else 4
        train_loader, val_loader, train_sampler = create_distributed_dataloaders(
            train_dataset, val_dataset, batch_size
        )
        
        # Training loop
        best_val_loss = float('inf')
        
        epoch_iterator = tqdm(
        range(epochs), 
        desc=f"Stage {stage_idx+1} Training", 
        disable=(rank != 0),
        unit="epoch"
        )

        if rank == 0:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    
        for epoch in epoch_iterator:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            
            # Capture metrics
            epoch_metrics = self._train_epoch(
                trainer, train_loader, embedding_cache, 
                device, epoch, stage_idx
            )
            
            epoch_loss = epoch_metrics['train_loss']
            stage_metrics['train_loss'].append(epoch_loss)
            
            print_main(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
            
            # Validation
            if (epoch + 1) % args.validation_interval == 0:
                val_metrics = self._validate_epoch(
                    trainer, val_loader, embedding_cache, device, stage_idx
                )
                
                val_loss = val_metrics['val_loss']
                stage_metrics['val_loss'].append(val_loss)
                stage_metrics['val_occupancy'].append(val_metrics.get('val_occupancy', 0))
                stage_metrics['val_density'].append(val_metrics.get('val_density', 0))

                if rank == 0:  # Only rank 0 generates visualizations
                    print_main("\nGenerating validation visualizations...")
                    
                    # Create sample descriptions
                    sample_descriptions = [
                        "A wooden chair with four legs",
                        "A simple ceramic mug",
                        "A spherical ball",
                        "A rectangular table"
                    ][:4]  # Adjust number of samples
                    
                    try:
                        self.visualizer.visualize_epoch_samples(
                            trainer=trainer,
                            epoch=epoch + 1,  # 1-indexed for display
                            descriptions=sample_descriptions,
                            embedding_cache=embedding_cache,
                            context_encoder=None,  # Not needed if cache works
                            tokenizer=None,
                            granularity=granularity,
                            device=device,
                            num_samples=len(sample_descriptions),
                            view_angles=[(30, 45), (60, 120), (0, 0)],  # Multiple views
                            show_slices=True,
                            show_progress=True
                        )
                        print_main("✓ Visualization saved successfully")
                    except Exception as e:
                        print_main(f"⚠ Visualization failed: {e}")

                # Update progress bar postfix instead of print
                if rank == 0:
                    epoch_iterator.set_postfix({
                        'loss': f'{epoch_loss:.4f}',
                        'best_val': f'{best_val_loss:.4f}'
                    })

                    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                    print(f"Peak Memory: {peak_memory:.2f} GB")
            
                # Synchronize after visualization
                if is_distributed:
                    dist.barrier()
                
                print_main(f"Validation Loss: {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print_main(f"✓ New best: {best_val_loss:.4f}")
        # Save stage metrics
        stage_metrics['best_val_loss'] = best_val_loss
        self.metrics_history['stages'][stage_idx] = stage_metrics
        
        # Save checkpoint for next stage
        self.save_stage_checkpoint(model, stage_idx, device)
        
        # Clean up
        del model
        del trainer
        torch.cuda.empty_cache()
        
        if is_distributed:
            dist.barrier()
        
        return stage_metrics
    
    def _train_epoch(self, trainer, dataloader, embedding_cache,
                device, epoch, stage_idx):
        """Training epoch with cached low-res contexts"""
        trainer.model.train()
        epoch_loss = 0
        num_batches = 0
        

        pbar = dataloader
        
        for batch_idx, (x0, descriptions) in enumerate(pbar):
            if is_distributed:
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                dtype = torch.float16
            x0 = x0.to(device, dtype=dtype)
            context = embedding_cache.get_batch_embeddings(descriptions, device)
            
            # Get precomputed low-res contexts
            low_res_batch = None
            if stage_idx > 0 and trainer.low_res_contexts:
                batch_start = batch_idx * dataloader.batch_size
                low_res_batch = []
                for i in range(len(descriptions)):
                    idx = batch_start + i
                    if idx in trainer.low_res_contexts:
                        low_res_batch.append(
                            trainer.low_res_contexts[idx].to(device, dtype=dtype)
                        )
                
                if low_res_batch:
                    low_res_batch = torch.stack(low_res_batch)
            
            # Training step
            loss = trainer.train_step(x0, context, low_res_batch)
            epoch_loss += loss
            num_batches += 1
            
            # if rank == 0:
            #     pbar.set_postfix({'loss': f'{loss:.4f}'})
        
        # Synchronize metrics across ranks
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
        """Validation epoch with comprehensive metrics"""
        trainer.model.eval()
        val_loss = 0
        total_occupancy = 0
        total_density = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (x0, descriptions) in enumerate(dataloader):
                if num_batches >= 10:  # Limit validation batches
                    break
                
                if is_distributed:
                    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                else:
                    dtype = torch.float16
                x0 = x0.to(device, dtype=dtype)
                context = embedding_cache.get_batch_embeddings(descriptions, device)
                
                # Get low-res contexts if available
                low_res_batch = None
                if stage_idx > 0 and trainer.low_res_contexts:
                    batch_start = batch_idx * dataloader.batch_size
                    low_res_batch = []
                    for i in range(len(descriptions)):
                        idx = batch_start + i
                        if idx in trainer.low_res_contexts:
                            low_res_batch.append(
                                trainer.low_res_contexts[idx].to(device, dtype=dtype)
                            )
                    
                    if low_res_batch:
                        low_res_batch = torch.stack(low_res_batch)
                
                # Compute loss
                batch_loss = trainer.compute_loss(x0, context, low_res_batch, backward=False)
                val_loss += batch_loss
                
                # Compute occupancy and density
                threshold = 0.5
                occupancy = (x0 > threshold).float().mean().item()
                density = x0.mean().item()
                total_occupancy += occupancy
                total_density += density
                
                num_batches += 1
        
        # Synchronize across ranks
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
    
    def visualize_epoch_samples(self, trainer, epoch, descriptions=None,
                            embedding_cache=None, context_encoder=None, 
                            tokenizer=None, granularity=32, device='cuda', 
                            num_samples=4, view_angles=[(30, 45), (60, 120)],
                            colormap='viridis', show_slices=True, show_progress=True):
        """
        Generate and visualize samples at a given epoch.
        
        Args:
            trainer: LayerXLayerDiffusionTrainerV3 instance
            epoch: Current epoch number
            descriptions: List of text descriptions (uses defaults if None)
            embedding_cache: T5EmbeddingCache instance
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
                    predicted_noise = trainer.model(x, t_batch, l, context, prev_layer_features)

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
                    
                    predicted_noise = trainer.model(
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
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=4 * (world_size if is_distributed else 1),
        pin_memory=True,
        drop_last=False,
        persistent_workers=True
    )
    
    return train_loader, val_loader, train_sampler

# %%
print_main("\n" + "="*60)
print_main("Starting Training from Scratch - Progressive Granularity")
print_main("="*60)
# %%
train_dataset_3d = VoxelDataset(
    npy_folder_path=f'/home/ad.msoe.edu/benzshawelt/Kedziora/kedziora_research/description_data_generation/objaverse_parallel/objaverse_voxels/{granularity}', 
    description_folder_path='/home/ad.msoe.edu/benzshawelt/Kedziora/kedziora_research/description_data_generation/objaverse_parallel/objaverse_descriptions', 
    granularity=granularity, 
    test_count=100,  # Use full training set
    enable_fuzzy_matching=args.enable_fuzzy_matching
)

val_dataset_3d = VoxelDataset(
    npy_folder_path=f'/home/ad.msoe.edu/benzshawelt/Kedziora/kedziora_research/description_data_generation/objaverse_parallel/objaverse_voxels/{granularity}', 
    description_folder_path='/home/ad.msoe.edu/benzshawelt/Kedziora/kedziora_research/description_data_generation/objaverse_parallel/objaverse_descriptions', 
    granularity=granularity, 
    test_count=100,  # Use 100 samples for validation
    enable_fuzzy_matching=args.enable_fuzzy_matching
)


batch_size = args.batch_size if args.batch_size is not None else 4  # Your default batch size

dataloader_3d_train, dataloader_3d_val, train_sampler = create_distributed_dataloaders(
    train_dataset_3d,
    val_dataset_3d,
    batch_size
)

print_main(f"Batch size per GPU: {batch_size}")
print_main(f"Effective batch size: {batch_size * world_size}")


print_main(f"Training dataset size: {len(dataloader_3d_train.dataset)}")
print_main(f"Validation dataset size: {len(dataloader_3d_val.dataset)}")

# %%


# Initialize T5
t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
t5_model = T5EncoderModel.from_pretrained('t5-base').to(device)
t5_model.eval()
print_main(f"T5 model loaded with {sum(p.numel() for p in t5_model.parameters()):,} parameters")

# ========== SETUP T5 EMBEDDING CACHE ==========
print_main("\n" + "="*60)
print_main("Setting up T5 Embedding Cache")
print_main("="*60)
embedding_cache = T5EmbeddingCache(cache_dir='./t5_cache', max_length=77)
embedding_cache.load_cache()  # Load existing cache if available

# Precompute embeddings for all descriptions in both train and val datasets
# Pass the datasets directly - they will be scanned for uncached descriptions
print_main("Precomputing embeddings for training dataset...")
embedding_cache.precompute_embeddings(
    texts=train_dataset_3d,  # Pass dataset directly
    tokenizer=t5_tokenizer,
    model=t5_model,
    device=device,
    batch_size=256
)

print_main("Precomputing embeddings for validation dataset...")
embedding_cache.precompute_embeddings(
    texts=val_dataset_3d,  # Pass dataset directly
    tokenizer=t5_tokenizer,
    model=t5_model,
    device=device,
    batch_size=256
)

train_dataset_3d.save_miss_report(output_path='train_description_miss_report.txt')
val_dataset_3d.save_miss_report(output_path='val_description_miss_report.txt')
print("Saved miss reports for datasets.")

# Optional: Free T5 model from GPU to save memory (embeddings are now cached)
t5_model = t5_model.cpu()
print_main("✓ T5 embeddings cached! Training will use cached embeddings for speed.\n")

# %%

n_epochs_3d = args.num_epochs
validation_interval = args.validation_interval

print_main("\n" + "="*60)
print_main("Starting Distributed Training")
print_main("="*60)
print_main(f"Total Epochs: {n_epochs_3d}")
print_main(f"Validation Interval: {validation_interval}")
print_main("="*60 + "\n")

# Create progressive trainer
progressive_trainer = FSDPProgressiveGranularityTrainer(
    granularities=[16, 32, 64]
)

# Train all stages
results = progressive_trainer.train_all_stages(
    epochs_per_stage=[20, 15, 15],  # Epochs for 16³, 32³, 64³
    device=device,
    embedding_cache=embedding_cache
)

# Extract results
metrics_history = results['metrics_history']
checkpoints = results['checkpoints']

print_main("\n" + "="*60)
print_main("Training Complete!")
print_main("="*60)

# Save metrics
if rank == 0:  # Only rank 0 saves
    with open('progressive_training_metrics.json', 'w') as f:
        # Convert any tensor values to floats for JSON serialization
        metrics_to_save = {
            'stages': {},
            'global_train_loss': metrics_history['global_train_loss'],
            'global_val_loss': metrics_history['global_val_loss'],
            'global_epochs': metrics_history['global_epochs']
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
    print("✓ Training metrics saved to: progressive_training_metrics.json")

# %%
# Plot comprehensive training curves
if rank == 0:  # Only rank 0 creates visualizations
    print("\nGenerating training curves...")
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Global training loss (all stages concatenated)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(metrics_history['global_train_loss'], alpha=0.7, linewidth=1)
    
    # Add vertical lines to separate stages
    epoch_boundaries = [0]
    for stage_idx in range(len(progressive_trainer.granularities)):
        epochs = epochs_per_stage[stage_idx]
        epoch_boundaries.append(epoch_boundaries[-1] + epochs)
    
    for i, boundary in enumerate(epoch_boundaries[1:-1], 1):
        ax1.axvline(boundary, color='red', linestyle='--', alpha=0.5, linewidth=2)
        # Label the stages
        mid_point = (epoch_boundaries[i-1] + epoch_boundaries[i]) / 2
        gran = progressive_trainer.granularities[i-1]
        ax1.text(mid_point, ax1.get_ylim()[1] * 0.95, f'{gran}³', 
                ha='center', fontsize=12, fontweight='bold')
    
    # Label final stage
    mid_point = (epoch_boundaries[-2] + epoch_boundaries[-1]) / 2
    gran = progressive_trainer.granularities[-1]
    ax1.text(mid_point, ax1.get_ylim()[1] * 0.95, f'{gran}³', 
            ha='center', fontsize=12, fontweight='bold')
    
    ax1.set_xlabel('Global Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Progressive Training Loss (All Stages)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Per-stage training losses (overlaid)
    ax2 = fig.add_subplot(gs[1, 0])
    colors = ['blue', 'green', 'orange', 'purple']
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
    
    # 3. Validation losses per stage
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
    
    # 4. Best validation loss per stage (bar chart)
    ax4 = fig.add_subplot(gs[1, 2])
    stage_names = [f"{metrics_history['stages'][i]['granularity']}³" 
                   for i in range(len(progressive_trainer.granularities))]
    best_losses = [metrics_history['stages'][i]['best_val_loss'] 
                   for i in range(len(progressive_trainer.granularities))]
    bars = ax4.bar(stage_names, best_losses, color=colors[:len(stage_names)], alpha=0.7)
    ax4.set_ylabel('Best Validation Loss', fontsize=10)
    ax4.set_title('Best Val Loss per Stage', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for bar, loss in zip(bars, best_losses):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 5. Validation occupancy
    ax5 = fig.add_subplot(gs[2, 0])
    for stage_idx, stage_data in metrics_history['stages'].items():
        gran = stage_data['granularity']
        val_epochs = list(range(args.validation_interval, 
                               len(stage_data['train_loss']) + 1, 
                               args.validation_interval))
        occupancy = stage_data['val_occupancy']
        if occupancy:  # Only plot if we have data
            ax5.plot(val_epochs, occupancy, marker='s', label=f'{gran}³', 
                    alpha=0.8, color=colors[stage_idx % len(colors)], linewidth=2)
    ax5.set_xlabel('Stage Epoch', fontsize=10)
    ax5.set_ylabel('Occupancy Rate', fontsize=10)
    ax5.set_title('Validation Voxel Occupancy', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Validation density
    ax6 = fig.add_subplot(gs[2, 1])
    for stage_idx, stage_data in metrics_history['stages'].items():
        gran = stage_data['granularity']
        val_epochs = list(range(args.validation_interval, 
                               len(stage_data['train_loss']) + 1, 
                               args.validation_interval))
        density = stage_data['val_density']
        if density:  # Only plot if we have data
            ax6.plot(val_epochs, density, marker='^', label=f'{gran}³', 
                    alpha=0.8, color=colors[stage_idx % len(colors)], linewidth=2)
    ax6.set_xlabel('Stage Epoch', fontsize=10)
    ax6.set_ylabel('Mean Density', fontsize=10)
    ax6.set_title('Validation Voxel Density', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Training summary statistics
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    
    summary_text = "Training Summary\n" + "="*30 + "\n\n"
    total_epochs = sum(epochs_per_stage)
    summary_text += f"Total Epochs: {total_epochs}\n"
    summary_text += f"Stages: {len(progressive_trainer.granularities)}\n\n"
    
    for stage_idx, gran in enumerate(progressive_trainer.granularities):
        stage_data = metrics_history['stages'][stage_idx]
        summary_text += f"Stage {stage_idx + 1} ({gran}³):\n"
        summary_text += f"  Epochs: {epochs_per_stage[stage_idx]}\n"
        summary_text += f"  Final Train Loss: {stage_data['train_loss'][-1]:.4f}\n"
        summary_text += f"  Best Val Loss: {stage_data['best_val_loss']:.4f}\n"
        summary_text += f"  Checkpoint: {checkpoints[stage_idx]}\n\n"
    
    ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Progressive Granularity Training - Comprehensive Metrics', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('progressive_training_curves.png', dpi=150, bbox_inches='tight')
    print("✓ Training curves saved to: progressive_training_curves.png")
    plt.show()
    plt.close()
    
    print("\n" + "="*60)
    print("All Done! 🎉")
    print("="*60)
    print("\nKey files saved:")
    for stage_idx, checkpoint_path in checkpoints.items():
        print(f"  - {checkpoint_path} (stage {stage_idx})")
    print("  - progressive_training_metrics.json (complete metrics history)")
    print("  - progressive_training_curves.png (training visualization)")
    print("  - ./t5_cache/ (cached T5 embeddings)")
    print("\n" + "="*60)

# Cleanup
if is_distributed:
    dist.destroy_process_group()
torch.cuda.empty_cache()