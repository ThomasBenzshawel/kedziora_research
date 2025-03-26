import math
from typing import Optional, Tuple

import torch

import fvdb
import fvdb.nn as fvnn
from fvdb.nn import VDBTensor
from inspect import isfunction
import math
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from fvdb.nn import VDBTensor
from fvdb import JaggedTensor, GridBatch
from fvdb.nn import GELU

from modules.diffusionmodules.util import checkpoint, conv_nd

def zero_module(module: nn.Module):
    """Zero out the parameters of a module."""
    for p in module.parameters():
        p.data.zero_()
    return module

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

class LayerNorm(nn.LayerNorm):
    def forward(self, input: VDBTensor) -> VDBTensor:
        num_channels = input.jdata.size(1)
        num_batches = input.grid.grid_count

        flat_data, flat_offsets = input.data.jdata, input.data.joffsets

        result_data = torch.empty_like(flat_data)

        for b in range(num_batches):
            feat = flat_data[flat_offsets[b]:flat_offsets[b +1]]
            if feat.size(0) != 0:
                feat = feat.reshape(1, -1, num_channels)
                feat = super().forward(feat)
                feat = feat.reshape(-1, num_channels)

                result_data[flat_offsets[b]:flat_offsets[b + 1]] = feat

        return VDBTensor(input.grid, input.grid.jagged_like(result_data), input.kmap)

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, data: VDBTensor):
        x = data.jdata
        x, gate = self.proj(x).chunk(2, dim=-1)
        out = x * F.gelu(gate)
        return VDBTensor(data.grid, data.grid.jagged_like(out), data.kmap)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            fvnn.Linear(dim, inner_dim),
            GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            fvnn.Dropout(dropout),
            fvnn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.scale = dim_head ** -0.5
        
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.heads = heads

        self.to_q = fvnn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = fvnn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = fvnn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            fvnn.Linear(inner_dim, query_dim),
            fvnn.Dropout(dropout)
        )

    def forward(self, x: VDBTensor):
        q = self.to_q(x)
        context = x
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Get flattened data and offsets
        q_data, q_offsets = q.data.jdata, q.data.joffsets
        k_data, k_offsets = k.data.jdata, k.data.joffsets
        v_data, v_offsets = v.data.jdata, v.data.joffsets
        
        num_batches = q.grid.grid_count
        result_data = torch.empty_like(q_data)
        
        for batch_idx in range(num_batches):
            # Extract batch data using offsets
            batch_q = q_data[q_offsets[batch_idx]:q_offsets[batch_idx + 1]]
            batch_k = k_data[k_offsets[batch_idx]:k_offsets[batch_idx + 1]]
            batch_v = v_data[v_offsets[batch_idx]:v_offsets[batch_idx + 1]]
            
            # Skip empty batches
            if batch_q.size(0) == 0:
                continue
                
            # Process this batch
            batch_out = self._attention(batch_q, batch_k, batch_v)
                
            # Store result back in the output tensor
            result_data[q_offsets[batch_idx]:q_offsets[batch_idx + 1]] = batch_out
        
        # Create a new VDBTensor with the original grid and processed data
        out = VDBTensor(x.grid, x.grid.jagged_like(result_data), x.kmap)
        return self.to_out(out)
    
    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # q: (N, C)
        # k: (77, C)
        # v: (77, C)
        # mask: (1, 77)
        h = self.heads
        q, k, v = map(lambda t: rearrange(t, '(b n) (h d) -> b h n d', h=h, b=1), (q, k, v))
        with torch.backends.cuda.sdp_kernel(enable_math=False):
            out = F.scaled_dot_product_attention(q, k, v)[0] # h, n, d
        out = rearrange(out, 'h n d -> n (h d)')
        return out
    

class OldAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.scale = dim_head ** -0.5
        
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads

        self.to_q = fvnn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = fvnn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = fvnn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            fvnn.Linear(inner_dim, query_dim),
            fvnn.Dropout(dropout)
        )

    def forward(self, x: VDBTensor):
        q = self.to_q(x)
        context = x
        k = self.to_k(context)
        v = self.to_v(context)
        
        out = []
        for batch_idx in range(q.grid.grid_count):
            out.append(self._attention(q[batch_idx].jdata, k[batch_idx].jdata, v[batch_idx].jdata))
        out = fvdb.JaggedTensor(out)
        out = VDBTensor(x.grid, out, x.kmap)
        return self.to_out(out)

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # q: (N, C)
        # k: (77, C)
        # v: (77, C)
        # mask: (1, 77)
        h = self.heads
        q, k, v = map(lambda t: rearrange(t, '(b n) (h d) -> b h n d', h=h, b=1), (q, k, v))
        with torch.backends.cuda.sdp_kernel(enable_math=False):
            out = F.scaled_dot_product_attention(q, k, v)[0] # h, n, d
        out = rearrange(out, 'h n d -> n (h d)')
        return out


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.scale = dim_head ** -0.5
        
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads

        self.to_q = fvnn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            fvnn.Linear(inner_dim, query_dim),
            fvnn.Dropout(dropout)
        )

    # def forward(self, x: VDBTensor, context: torch.Tensor=None, mask=None):
    #     q = self.to_q(x)
    #     k = self.to_k(context)
    #     v = self.to_v(context)
        
    #     out = []
        
    #     for batch_idx in range(q.grid.grid_count):
    #         if exists(mask):
    #             mask = rearrange(mask, 'b ... -> b (...)')

    #             out.append(self._attention(q[batch_idx].jdata, k[batch_idx], v[batch_idx], mask=mask[batch_idx:batch_idx+1]))
    #         else:
    #             out.append(self._attention(q[batch_idx].jdata, k[batch_idx], v[batch_idx]))
    #     out = fvdb.JaggedTensor(out)
    #     out = VDBTensor(x.grid, out, x.kmap)
    #     return self.to_out(out)

    def forward(self, x: VDBTensor, context: torch.Tensor=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Get flattened data and offsets
        q_data, q_offsets = q.data.jdata, q.data.joffsets
        num_batches = q.grid.grid_count
        
        out_data = torch.empty_like(q_data)
        
        for batch_idx in range(num_batches):
            # Extract batch data using offsets
            batch_q = q_data[q_offsets[batch_idx]:q_offsets[batch_idx + 1]]
            
            if batch_q.size(0) == 0:
                continue
                
            if exists(mask):
                batch_mask = mask[batch_idx:batch_idx+1]
                batch_out = self._attention(batch_q, k[batch_idx], v[batch_idx], mask=batch_mask)
            else:
                batch_out = self._attention(batch_q, k[batch_idx], v[batch_idx])
                
            # Store result back in the output tensor
            out_data[q_offsets[batch_idx]:q_offsets[batch_idx + 1]] = batch_out
        
        # Create a new VDBTensor with the original grid and processed data
        out = VDBTensor(x.grid, x.grid.jagged_like(out_data), x.kmap)
        return self.to_out(out)

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
        # q: (N, C)
        # k: (77, C)
        # v: (77, C)
        # mask: (1, 77)
        if q.size(0) == 0:
            return q

        h = self.heads
        q, k, v = map(lambda t: rearrange(t, '(b n) (h d) -> b h n d', h=h, b=1), (q, k, v))
        
        if exists(mask):
            mask = repeat(mask, 'b s -> b h l s', h=h, l=q.shape[2])
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)[0] # h, n, d
        else:
            with torch.backends.cuda.sdp_kernel(enable_math=False):
                out = F.scaled_dot_product_attention(q, k, v)[0] # h, n, d
        out = rearrange(out, 'h n d -> n (h d)')
        return out

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True, disable_sa=False):
        super().__init__()
        if not disable_sa:
            self.attn1 = Attention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        if not disable_sa:
            self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.norm3 = LayerNorm(dim)
        self.checkpoint = checkpoint
        self.disable_sa = disable_sa

    def forward(self, x, context=None, mask=None):
        if not self.disable_sa:        
            x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context, mask=mask) + x
        x = self.ff(self.norm3(x)) + x
        return x

class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, disable_sa=False):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = fvnn.GroupNorm(32, in_channels)

        self.proj_in = fvnn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim, disable_sa=disable_sa)
                for d in range(depth)]
        )

        self.proj_out = zero_module(fvnn.Linear(inner_dim, in_channels))

    def forward(self, x: VDBTensor, context=None, mask=None):
        # if x is empty
        if x.grid.ijk.jdata.size(0) == 0:
            return x
        
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        for block in self.transformer_blocks:
            x = block(x, context=context, mask=mask)
        x = self.proj_out(x)
        return x + x_in
    

class TimestepModule(nn.Module):
    def forward(self, x, emb, target_tensor = None):
        raise NotImplementedError


class TimestepSequential(nn.Sequential):
    def forward(self, x, emb, target_tensor: Optional[VDBTensor] = None, context=None, mask=None):
        for layer in self:
            if isinstance(layer, TimestepModule):
                x = layer(x, emb, target_tensor)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context, mask)
            else:
                x = layer(x)
        return x


class ResBlock(TimestepModule):
    def __init__(self, channels: int, emb_channels: int, dropout: float,
                 out_channels: Optional[int] = None,
                 up: bool = False, down: bool = False, stride: int = 1):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.stride = stride

        # Nonlinear operations to time/class embeddings
        #   (added between in_layers and out_layers in the res branch)
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.emb_channels, 2 * self.out_channels)
        )

        self.in_layers = nn.Sequential(
            fvnn.GroupNorm(num_groups=32, num_channels=channels),
            fvnn.SiLU(),
            fvnn.SparseConv3d(self.channels, self.out_channels, 3, bias=True)
        )

        self.up, self.down = up, down
        if self.up:
            self.up_module = fvnn.UpsamplingNearest(self.stride)
        elif self.down:
            self.down_module = fvnn.AvgPool(self.stride)

        self.out_layers = nn.Sequential(
            fvnn.GroupNorm(num_groups=32, num_channels=self.out_channels),
            fvnn.SiLU(),
            fvnn.Dropout(p=self.dropout),
            # Zero out res output since this is the residual
            zero_module(fvnn.SparseConv3d(self.out_channels, self.out_channels, 3, bias=True))
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = fvnn.SparseConv3d(channels, self.out_channels, 1, bias=True)

    def forward(self, data: VDBTensor, emb: torch.Tensor,
                target_tensor: Optional[VDBTensor] = None):
        if self.up or self.down:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            data_h = in_rest(data)
            data_h = self.up_module(data_h, ref_fine_data=target_tensor) \
                if self.up else self.down_module(data_h, ref_coarse_data=target_tensor)
            data_h = in_conv(data_h)
            data = self.up_module(data, ref_fine_data=data_h) \
                if self.up else self.down_module(data, ref_coarse_data=data_h)
        else:
            data_h = self.in_layers(data)

        assert isinstance(data_h, VDBTensor)

        emb_h = self.emb_layers(emb)    # (B, 2C)
        scale, shift = emb_h.chunk(2, dim=-1)   # (B, C), (B, C)
        batch_idx = data_h.jidx.long()

        out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        data_h = out_norm(data_h) * (1 + scale[batch_idx]) + shift[batch_idx]
        data_h = out_rest(data_h)

        data = data_h + self.skip_connection(data)
        return data

class UNetModel(nn.Module):
    def __init__(self, 
                 num_input_channels: int, 
                 model_channels: int, 
                 num_res_blocks: int,
                 out_channels: Optional[int] = None, 
                 dropout: float = 0.0,
                 channel_mult: Tuple = (1, 2, 4, 8), 
                 num_classes: Optional[int] = None, 
                 attention_resolutions: list = [],
                 num_heads: int = 8,
                 transformer_depth: int = 1,
                 context_dim: int = 1024,
                 **kwargs):
        super().__init__()

        in_channels = num_input_channels
        self.in_channels = in_channels
        self.model_channels = model_channels

        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        transformer_depth_middle = transformer_depth[-1]

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "provide num_res_blocks either as an int (globally constant) or "
                    "as a list/tuple (per-level) with the same length as channel_mult"
                )
            self.num_res_blocks = num_res_blocks

        self.out_channels = out_channels or in_channels
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_classes = num_classes
        
        self.attention_resolutions = attention_resolutions
        self.num_heads = num_heads

        time_emb_dim = 4 * self.model_channels
        self.time_emb = nn.Sequential(
            nn.Linear(self.model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        if self.num_classes is not None:
            self.label_emb = nn.Linear(self.num_classes, time_emb_dim)

        # Encoder
        self.encoder_blocks = nn.ModuleList([TimestepSequential(
            fvnn.SparseConv3d(self.in_channels, self.model_channels, 3, bias=True),
        )])

        encoder_channels = [self.model_channels]
        current_channels = self.model_channels
        ds = 1
        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks[level]):
                layers = [ResBlock(
                    current_channels, time_emb_dim, self.dropout,
                    out_channels=self.model_channels * mult
                )]
                current_channels = self.model_channels * mult
                # Attention
                if ds in attention_resolutions:
                    # enable self-attention
                    disable_sa = False
                else:
                    disable_sa = True
                dim_head = current_channels // num_heads
                layers.append(
                    SpatialTransformer(current_channels, num_heads, dim_head, depth=transformer_depth[level], context_dim=context_dim, disable_sa=disable_sa)
                    )
                self.encoder_blocks.append(TimestepSequential(*layers))
                encoder_channels.append(current_channels)
            # Downsample for all but the last block
            if level < len(self.channel_mult) - 1:
                layers = [ResBlock(
                    current_channels, time_emb_dim, self.dropout,
                    out_channels=current_channels,
                    down=True, stride=2
                )]
                self.encoder_blocks.append(TimestepSequential(*layers))
                encoder_channels.append(current_channels)
                ds *= 2

        # Middle block (won't change dimension)
        self.middle_block = TimestepSequential(
            ResBlock(current_channels, time_emb_dim, self.dropout),
            SpatialTransformer(current_channels, num_heads, dim_head, depth=transformer_depth_middle, context_dim=context_dim, disable_sa=False),
            ResBlock(current_channels, time_emb_dim, self.dropout)
        )

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            # Use one more block for decoder
            for i in range(self.num_res_blocks[level] + 1):
                skip_channels = encoder_channels.pop()
                layers = [ResBlock(
                    current_channels + skip_channels,
                    time_emb_dim, self.dropout,
                    out_channels=self.model_channels * mult
                )]
                current_channels = self.model_channels * mult
                # Attention
                if ds in attention_resolutions:
                    # enable self-attention
                    disable_sa = False
                else:
                    disable_sa = True
                layers.append(
                    SpatialTransformer(current_channels, num_heads, dim_head, depth=transformer_depth[level], context_dim=context_dim, disable_sa=disable_sa)
                    )
                # Upsample for all but the finest block
                if level > 0 and i == self.num_res_blocks[level]:
                    layers.append(ResBlock(
                        current_channels, time_emb_dim, self.dropout,
                        out_channels=current_channels,
                        up=True, stride=2
                    ))
                    ds //= 2
                self.decoder_blocks.append(TimestepSequential(*layers))

        # Output block
        assert current_channels == self.model_channels
        self.out_block = nn.Sequential(
            fvnn.GroupNorm(num_groups=32, num_channels=current_channels),
            fvnn.SiLU(),
            zero_module(fvnn.SparseConv3d(current_channels, self.out_channels, 3, bias=True))
        )

    def timestep_encoding(self, timesteps: torch.Tensor, max_period: int = 10000):
        dim = self.model_channels
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, data: VDBTensor, timesteps: torch.Tensor,
                y: Optional[torch.Tensor] = None, context: torch.Tensor = None, mask: torch.Tensor = None):
        assert (y is not None) == (self.num_classes is not None), \
            "Must provide labels if num_classes is not None"
        if timesteps.dim() == 0:
            timesteps = timesteps.expand(1).repeat(data.grid.grid_count).to(data.device)
        
        t_emb = self.timestep_encoding(timesteps)
        emb = self.time_emb(t_emb)
        if y is not None:
            emb += self.label_emb(y)

        hs = []
        for block in self.encoder_blocks:

            data = block(data, emb, context=context, mask=mask)
            hs.append(data)
        data = self.middle_block(data, emb, context=context, mask=mask)
        for block in self.decoder_blocks:
            pop_data = hs.pop()
            data = fvdb.jcat([pop_data, data], dim=1)
            data = block(data, emb, hs[-1] if len(hs) > 0 else None, context=context, mask=mask)

        data = self.out_block(data)
        return data