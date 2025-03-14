import torch
import torch.nn as nn
import fvdb
from fvdb import JaggedTensor, GridBatch

from utils.Dataspec import DatasetSpec as DS

class Embedder(nn.Module):
    def __init__(self, include_input=True, input_dims=3, max_freq_log2=10, num_freqs=10, log_sampling=True, periodic_fns=[torch.sin, torch.cos]):
        super().__init__()
        embed_fns = []
        d = input_dims
        out_dim = 0
        if include_input:
            out_dim += d
            
        max_freq = max_freq_log2
        N_freqs = num_freqs
        
        if log_sampling:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
        
        for freq in freq_bands:
            for _ in periodic_fns:
                out_dim += d
        
        self.include_input = include_input
        self.freq_bands = freq_bands
        self.periodic_fns = periodic_fns
        
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def forward(self, inputs):
        output_list = [inputs]
        for fn in self.embed_fns:
            output_list.append(fn(inputs))
            
        for freq in self.freq_bands:
            for p_fn in self.periodic_fns:
                output_list.append(p_fn(inputs * freq))
        
        return torch.cat(output_list, -1)

def get_embedder(multires, i=0, input_dims=3):
    if i == -1:
        return nn.Identity(), 3
    
    embedder_obj = Embedder(max_freq_log2=multires-1, num_freqs=multires, input_dims=input_dims)
    return embedder_obj, embedder_obj.out_dim
     
class Encoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        encoder_input_dims = 3
        if self.hparams.use_input_normal:
            encoder_input_dims += 3
        if self.hparams.use_input_semantic:
            encoder_input_dims += self.hparams.dim_semantic
            self.semantic_embed_fn = nn.Embedding(self.hparams.num_semantic, self.hparams.dim_semantic)
        if self.hparams.use_input_color:
            encoder_input_dims += 3

        embed_fn, input_ch = get_embedder(5)
        self.pos_embedder = embed_fn
        
        input_dim = 0
        input_dim += input_ch
        if self.hparams.use_input_normal:
            input_dim += 3 # normal
        if self.hparams.use_input_intensity:
            input_dim += 1
        if self.hparams.use_input_semantic:
            input_dim += self.hparams.dim_semantic
        if self.hparams.use_input_color:
            input_dim += 3 # color
            
        self.mix_fc = nn.Linear(input_dim, self.hparams.network.encoder.c_dim)

    


    def forward(self, grid: GridBatch, batch) -> torch.Tensor:
        input_normal = batch[DS.TARGET_NORMAL] if DS.TARGET_NORMAL in batch.keys() else None
        if self.hparams.use_input_color:
            input_color = batch[DS.INPUT_COLOR]
        else:
            input_color = None          

        coords = grid.grid_to_world(grid.ijk.float()).jdata
        unet_feat = self.pos_embedder(coords)
        
        if self.hparams.use_input_normal:
            ref_grid = batch['input_grid']
            ref_xyz = ref_grid.grid_to_world(ref_grid.ijk.float()) 
            # splatting normal
            input_normal = grid.splat_trilinear(ref_xyz, fvdb.JaggedTensor(input_normal))
            # normalize normal
            input_normal.jdata /= (input_normal.jdata.norm(dim=1, keepdim=True) + 1e-6)
            unet_feat = torch.cat([unet_feat, input_normal.jdata], dim=1)

        if self.hparams.use_input_semantic:
            input_semantic = fvdb.JaggedTensor(batch[DS.GT_SEMANTIC])
            input_semantic_embed = self.semantic_embed_fn(input_semantic.jdata.long())
            unet_feat = torch.cat([unet_feat, input_semantic_embed], dim=1)

        if self.hparams.use_input_intensity:
            input_intensity = fvdb.JaggedTensor(batch[DS.INPUT_INTENSITY])
            unet_feat = torch.cat([unet_feat, input_intensity.jdata], dim=1)
            
        if self.hparams.use_input_color:
            input_color = fvdb.JaggedTensor(batch[DS.INPUT_COLOR])
            unet_feat = torch.cat([unet_feat, input_color.jdata], dim=1)

        unet_feat = self.mix_fc(unet_feat)
        return unet_feat