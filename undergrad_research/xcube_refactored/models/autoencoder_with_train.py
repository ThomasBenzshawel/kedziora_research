# This file is a refactored version of the original code from the following repository:

# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.


# Python standard library
import functools
import gc
import importlib
import inspect
import multiprocessing
import pickle
import shutil
import traceback
from collections import OrderedDict, defaultdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union

# Third-party libraries - NumPy & Scientific
import numpy as np
from numpy.random import RandomState

# Third-party libraries - PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard.summary import hparams

# Third-party libraries - Visualization
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Third-party libraries - ML Tools
import pytorch_lightning as pl
import wandb
from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf
import omegaconf.errors

# Local imports
from ext import common
import fvdb
import fvdb.nn as fvnn
from fvdb import JaggedTensor, GridBatch


###############################THINGS I ACTUALLY SUPPORT BEING OUTSIDE THIS FILE############################
from xcube_refactored.modules.autoencoding.hparams import hparams_handler
from xcube_refactored.utils.loss_util import AverageMeter
from xcube_refactored.utils.loss_util import TorchLossMeter
############################################################################################################

############################################################################################################
from xcube_refactored.modules.autoencoding.sunet import StructPredictionNet # TODO: THIS IS VERY IMPORTANT BUT CONTAINS 500 LINES OF CODE
############################################################################################################

############################################################################################################
from xcube_refactored.utils import exp # 1000 lines of code that are experiment utilities
############################################################################################################


def color_from_points(target_pcs, ref_pcs, ref_colors, k=8):
    """
    Compute the color of each point in the target point cloud by weighted average of the colors of its k nearest neighbors in the reference point cloud.

    Parameters:
        target_pcs (torch.Tensor): Coordinates of the points in the target point cloud, size (N, 3).
        ref_pcs (torch.Tensor): Coordinates of the points in the reference point cloud, size (M, 3).
        ref_colors (torch.Tensor): Colors of each point in the reference point cloud, size (M, 3).
        k (int): Number of nearest neighbors to consider for each point in the target point cloud.

    Returns:
        torch.Tensor: Calculated colors for each point in the target point cloud, size (N, 3).
    """
    if target_pcs.shape[0] == 0:
        return torch.zeros((0, 3), dtype=torch.float32, device=target_pcs.device)
    torch.cuda.empty_account()
    dist, idx = common.knn_query_fast(target_pcs.contiguous(), ref_pcs.contiguous(), k)
    dist = dist.sqrt()

    knn_color = ref_colors[idx.long()]
    weight = 1 / (dist + 1e-8)  # N, K, inverse distance weighting
    weight = weight / weight.sum(dim=1, keepdim=True)  # normalize weights across each point's k neighbors
    target_color = (weight.unsqueeze(-1) * knn_color).sum(dim=1)  # weighted average of colors

    return target_color

def semantic_from_points(target_pcs, ref_pcs, ref_semantic):
    if target_pcs.shape[0] == 0:
        return torch.zeros((0), dtype=torch.int64, device=target_pcs.device)
    torch.cuda.empty_cache()
    dist, idx = common.knn_query_fast(target_pcs.contiguous(), ref_pcs.contiguous(), 1)
    dist = dist.sqrt()

    knn_color = ref_semantic[idx.long()]
    return knn_color[:, 0].long()

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
    
class Loss(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

    def transform_field(self, field: torch.Tensor):
        gt_band = 1.0 # not sure if this will be changed
        truncation_size = gt_band * self.hparams.voxel_size
        # non-binary supervision (made sure derivative norm at 0 if 1)
        field = torch.tanh(field / truncation_size) * truncation_size
        return field
    
    def cross_entropy(self, pd_struct: fvnn.VDBTensor, gt_grid: fvdb.GridBatch, dynamic_grid: fvdb.GridBatch = None):
        assert torch.allclose(pd_struct.grid.origins, gt_grid.origins)
        assert torch.allclose(pd_struct.grid.voxel_sizes, gt_grid.voxel_sizes)
        idx_mask = gt_grid.ijk_to_index(pd_struct.grid.ijk).jdata == -1
        idx_mask = idx_mask.long()
        if dynamic_grid is not None:
            dynamic_mask = dynamic_grid.ijk_to_index(pd_struct.grid.ijk).jdata == -1
            loss = F.cross_entropy(pd_struct.feature.jdata, idx_mask, reduction='none') * dynamic_mask.float()
            loss = loss.mean()
        else:
            loss = F.cross_entropy(pd_struct.feature.jdata, idx_mask)
        return 0.0 if idx_mask.size(0) == 0 else loss
    
    def struct_acc(self, pd_struct: fvnn.VDBTensor, gt_grid: fvdb.GridBatch):
        assert torch.allclose(pd_struct.grid.origins, gt_grid.origins)
        assert torch.allclose(pd_struct.grid.voxel_sizes, gt_grid.voxel_sizes)
        idx_mask = gt_grid.ijk_to_index(pd_struct.grid.ijk).jdata == -1
        idx_mask = idx_mask.long()
        return torch.mean((pd_struct.feature.jdata.argmax(dim=1) == idx_mask).float())
    
    def grid_iou(self, gt_grid: fvdb.GridBatch, pd_grid: fvdb.GridBatch):
        assert gt_grid.grid_count == pd_grid.grid_count
        idx = pd_grid.ijk_to_index(gt_grid.ijk)
        upi = (pd_grid.num_voxels + gt_grid.num_voxels).cpu().numpy().tolist()
        ious = []
        for i in range(len(upi)):
            inter = torch.sum(idx[i].jdata >= 0).item()
            ious.append(inter / (upi[i] - inter + 1.0e-6))
        return np.mean(ious)

    def normal_loss(self, batch, normal_feats: fvnn.VDBTensor, eps=1e-6):
        if self.hparams.use_fvdb_loader:
            ref_grid = batch['input_grid']
            ref_xyz = ref_grid.grid_to_world(ref_grid.ijk.float()) 
        else:
            ref_xyz = fvdb.JaggedTensor(batch[DS.INPUT_PC])
        
        gt_normal = normal_feats.grid.splat_trilinear(ref_xyz, fvdb.JaggedTensor(batch[DS.TARGET_NORMAL]))
        # normalize normal
        gt_normal.jdata /= (gt_normal.jdata.norm(dim=1, keepdim=True) + eps)
        normal_loss = F.l1_loss(gt_normal.jdata, normal_feats.feature.jdata)
        return normal_loss
    
    def color_loss(self, batch, color_feats: fvnn.VDBTensor):
        assert self.hparams.use_fvdb_loader is True
        # check if color_feats is empty
        if color_feats.grid.total_voxels == 0:
            return 0.0
        ref_grid = batch['input_grid']
        ref_xyz = ref_grid.grid_to_world(ref_grid.ijk.float())
        ref_color = fvdb.JaggedTensor(batch[DS.INPUT_COLOR])
        
        target_xyz = color_feats.grid.grid_to_world(color_feats.grid.ijk.float())
        target_color = []
        slect_color_feats = []
        for batch_idx in range(ref_grid.grid_count):
            ref_color_i = ref_color[batch_idx].jdata
            target_color.append(color_from_points(target_xyz[batch_idx].jdata, ref_xyz[batch_idx].jdata, ref_color_i, k=1))
            slect_color_feats.append(color_feats.feature[batch_idx].jdata)
            
        if len(target_color) == 0 or len(slect_color_feats) == 0: # to avoid JaggedTensor build from empty list
            return 0.0  
        
        target_color = fvdb.JaggedTensor(target_color)
        slect_color_feats = fvdb.JaggedTensor(slect_color_feats)
        color_loss = F.l1_loss(slect_color_feats.jdata, target_color.jdata)
        return color_loss
    
    def semantic_loss(self, batch, semantic_feats: fvnn.VDBTensor):
        assert self.hparams.use_fvdb_loader is True
        # check if semantic_feats is empty
        if semantic_feats.grid.total_voxels == 0:
            return 0.0
        ref_grid = batch['input_grid']
        ref_xyz = ref_grid.grid_to_world(ref_grid.ijk.float())
        ref_semantic = fvdb.JaggedTensor(batch[DS.GT_SEMANTIC])
        if ref_semantic.jdata.size(0) == 0: # if all samples in this batch is without semantic
            return 0.0
                
        target_xyz = semantic_feats.grid.grid_to_world(semantic_feats.grid.ijk.float())       
        target_semantic = []
        slect_semantic_feats = []
        for batch_idx in range(ref_grid.grid_count):
            ref_semantic_i = ref_semantic[batch_idx].jdata
            if ref_semantic_i.size(0) == 0:
                continue
            target_semantic.append(semantic_from_points(target_xyz[batch_idx].jdata, ref_xyz[batch_idx].jdata, ref_semantic_i))
            slect_semantic_feats.append(semantic_feats.feature[batch_idx].jdata)
                    
        if len(target_semantic) == 0 or len(slect_semantic_feats) == 0: # to avoid JaggedTensor build from empty list
            return 0.0

        target_semantic = fvdb.JaggedTensor(target_semantic)
        slect_semantic_feats = fvdb.JaggedTensor(slect_semantic_feats)
        
        if slect_semantic_feats.jdata.size(0) == 0: # to aviod cross_entropy take empty tensor
            return 0.0
        
        semantic_loss = F.cross_entropy(slect_semantic_feats.jdata, target_semantic.jdata.long())
        return semantic_loss
    
    def get_kl_weight(self, global_step):
        # linear annealing the kl weight
        if global_step > self.hparams.anneal_star_iter:
            if global_step < self.hparams.anneal_end_iter:
                kl_weight = self.hparams.kl_weight_min + \
                                         (self.hparams.kl_weight_max - self.hparams.kl_weight_min) * \
                                         (global_step - self.hparams.anneal_star_iter) / \
                                         (self.hparams.anneal_end_iter - self.hparams.anneal_star_iter)
            else:
                kl_weight = self.hparams.kl_weight_max
        else:
            kl_weight = self.hparams.kl_weight_min

        return kl_weight

    def forward(self, batch, out, compute_metric: bool, global_step, current_epoch, optimizer_idx=0):
        loss_dict = TorchLossMeter()
        metric_dict = TorchLossMeter()
        latent_dict = TorchLossMeter()

        dynamic_grid = None

        if not self.hparams.use_hash_tree:
            gt_grid = out['gt_grid']
            if self.hparams.supervision.structure_weight > 0.0:
                for feat_depth, pd_struct_i in out['structure_features'].items():
                    downsample_factor = 2 ** feat_depth
                    if self.hparams.remain_h:
                        pd_voxel_size = pd_struct_i.grid.voxel_sizes[0]
                        h_factor = pd_voxel_size[0] // pd_voxel_size[2]
                        downsample_factor = [downsample_factor, downsample_factor, downsample_factor // h_factor]
                    if downsample_factor != 1:             
                        gt_grid_i = gt_grid.coarsened_grid(downsample_factor)
                        dyn_grid_i = dynamic_grid.coarsened_grid(downsample_factor) if dynamic_grid is not None else None
                    else:
                        gt_grid_i = gt_grid
                        dyn_grid_i = dynamic_grid
                    loss_dict.add_loss(f"struct-{feat_depth}", self.cross_entropy(pd_struct_i, gt_grid_i, dyn_grid_i),
                                    self.hparams.supervision.structure_weight)
                    if compute_metric:
                        with torch.no_grad():
                            metric_dict.add_loss(f"struct-acc-{feat_depth}", self.struct_acc(pd_struct_i, gt_grid_i))
        else:
            if self.hparams.supervision.structure_weight > 0.0:
                gt_tree = out['gt_tree']
                for feat_depth, pd_struct_i in out['structure_features'].items():
                    gt_grid_i = gt_tree[feat_depth]
                    # get dynamic grid
                    dyn_grid_i = dynamic_grid.coarsened_grid(2 ** feat_depth) if dynamic_grid is not None else None
                    loss_dict.add_loss(f"struct-{feat_depth}", self.cross_entropy(pd_struct_i, gt_grid_i, dyn_grid_i),
                                    self.hparams.supervision.structure_weight)
                    if compute_metric:
                        with torch.no_grad():
                            metric_dict.add_loss(f"struct-acc-{feat_depth}", self.struct_acc(pd_struct_i, gt_grid_i))
        
        # compute normal loss
        if self.hparams.with_normal_branch:
            if out['normal_features'] == {}:
                normal_loss = 0.0
            else:
                feat_depth = min(out['normal_features'].keys())
                normal_loss = self.normal_loss(batch, out['normal_features'][feat_depth])
                    
            loss_dict.add_loss(f"normal", normal_loss, self.hparams.supervision.normal_weight)
        
        # compute semantic loss
        if self.hparams.with_semantic_branch:
            for feat_depth, pd_semantic_i in out['semantic_features'].items():
                semantic_loss = self.semantic_loss(batch, pd_semantic_i)
                if semantic_loss == 0.0: # do not take empty into log
                    continue
                loss_dict.add_loss(f"semantic_{feat_depth}", semantic_loss, self.hparams.supervision.semantic_weight)
                
        # compute color loss
        if self.hparams.with_color_branch:
            for feat_depth, pd_color_i in out['color_features'].items():
                color_loss = self.color_loss(batch, pd_color_i)
                if color_loss == 0.0:
                    continue
                loss_dict.add_loss(f"color_{feat_depth}", color_loss, self.hparams.supervision.color_weight)

        # compute KL divergence
        if "dist_features" in out:
            dist_features = out['dist_features']
            kld = 0.0
            for latent_id, (mu, logvar) in enumerate(dist_features):
                num_voxel = mu.size(0)
                kld_temp = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kld_total = kld_temp.item()
                if self.hparams.normalize_kld:
                    kld_temp /= num_voxel

                kld += kld_temp
                latent_dict.add_loss(f"mu-{latent_id}", mu.mean())
                latent_dict.add_loss(f"logvar-{latent_id}", logvar.mean())
                latent_dict.add_loss(f"kld-true-{latent_id}", kld_temp.item())
                latent_dict.add_loss(f"kld-total-{latent_id}", kld_total)

            if self.hparams.enable_anneal:
                loss_dict.add_loss("kld", kld, self.get_kl_weight(global_step))
            else:
                loss_dict.add_loss("kld", kld, self.hparams.kl_weight)
            
        return loss_dict, metric_dict, latent_dict

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

def lambda_lr_wrapper(it, lr_config, batch_size):
    return max(
        lr_config['decay_mult'] ** (int(it * batch_size / lr_config['decay_step'])),
        lr_config['clip'] / lr_config['init'])

exp.global_var_manager.register_variable('skip_backward', False)

class DatasetSpec(Enum):
    SHAPE_NAME = 100
    INPUT_PC = 200
    TARGET_NORMAL = 300
    INPUT_COLOR = 350
    INPUT_INTENSITY = 360
    GT_DENSE_PC = 400
    GT_DENSE_NORMAL = 500
    GT_DENSE_COLOR = 550
    GT_MESH = 600
    GT_MESH_SOUP = 650
    GT_ONET_SAMPLE = 700
    GT_GEOMETRY = 800
    DATASET_CFG = 1000
    GT_DYN_FLAG = 1100
    GT_SEMANTIC = 1200
    LATENT_SEMANTIC = 1300
    SINGLE_SCAN_CROP = 1400
    SINGLE_SCAN_INTENSITY_CROP = 1410
    SINGLE_SCAN = 1450
    SINGLE_SCAN_INTENSITY = 1460
    CLASS = 1500
    TEXT_EMBEDDING = 1600
    TEXT_EMBEDDING_MASK = 1610
    TEXT = 1620
    MICRO = 1630

class RandomSafeDataset(Dataset):

    def __init__(self, seed: int, _is_val: bool = False, skip_on_error: bool = False):
        self._seed = seed
        self._is_val = _is_val
        self.skip_on_error = skip_on_error
        if not self._is_val:
            self._manager = multiprocessing.Manager()
            self._read_count = self._manager.dict()
            self._rc_lock = multiprocessing.Lock()

    def get_rng(self, idx):
        if self._is_val:
            return RandomState(self._seed)
        with self._rc_lock:
            if idx not in self._read_count:
                self._read_count[idx] = 0
            rng = RandomState(exp.deterministic_hash((idx, self._read_count[idx], self._seed)))
            self._read_count[idx] += 1
        return rng

    def sanitize_specs(self, old_spec, available_spec):
        old_spec = set(old_spec)
        available_spec = set(available_spec)
        for os in old_spec:
            assert isinstance(os, DatasetSpec)
        new_spec = old_spec.intersection(available_spec)
        # lack_spec = old_spec.difference(new_spec)
        # if len(lack_spec) > 0:
        #     exp.logger.warning(f"Lack spec {lack_spec}.")
        return new_spec

    def _get_item(self, data_id, rng):
        raise NotImplementedError

    def __getitem__(self, data_id):
        rng = self.get_rng(data_id)
        if self.skip_on_error:
            try:
                return self._get_item(data_id, rng)
            except ConnectionAbortedError:
                return self.__getitem__(rng.randint(0, len(self) - 1))
            except Exception:
                # Just return a random other item.
                print(f"Warning: Get item {data_id} error, but handled.")
                return self.__getitem__(rng.randint(0, len(self) - 1))
        else:
            try:
                return self._get_item(data_id, rng)
            except ConnectionAbortedError:
                return self.__getitem__(rng.randint(0, len(self) - 1))


def list_collate(batch):
    """
    This just do not stack batch dimension.
    """
    from fvdb import GridBatch, JaggedTensor

    elem = None
    for e in batch:
        if e is not None:
            elem = e
            break
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        return batch
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            return list_collate([torch.as_tensor(b) if b is not None else None for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, DictConfig) or isinstance(elem, ListConfig):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: list_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [list_collate(samples) for samples in transposed]
    elif isinstance(elem, GridBatch):
        return fvdb.cat(batch)
    
    # elif isinstance(elem, pathlib.Path):
    #     return batch
    # elif elem is None:
    #     return batch

    # raise NotImplementedError
    return batch

class BaseModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.best_metrics = AverageMeter()
        # step -> log_name -> log_value (list of ordered-dict)
        self.test_logged_values = []
        self.record_folder = None
        self.record_headers = []
        self.record_data_cache = {}
        self.last_test_valid = False
        self.render_backend = 'pyrender'
        self.num_oom = 0

    @staticmethod
    def load_module(spec_path, weight_path=None, overwrite_config=None):
        """
        Load a module given spec_path
        :param spec_path: Path to a model yaml file or ckpt. If is a ckpt file, then weight will also be loaded.
        :param weight_path: Path to the model weight. If explicitly set to 'NO_LOAD', then even if ckpt is provided to
            spec_path, no weights will be loaded into the model.
        :param overwrite_config: argparse.Namespace object, if you want to overwrite the original config.
        :return: the module class, possibly with weight loaded.
        """
        if spec_path is not None:
            spec_path = Path(spec_path)
            if spec_path.suffix == ".ckpt":
                # Boil down spec path using glob.
                import glob2
                possible_paths = glob2.glob(str(spec_path))
                if len(possible_paths) == 1:
                    spec_path = Path(possible_paths[0])
                else:
                    raise AssertionError
                config_yaml_path = spec_path.parent.parent / "hparams.yaml"
                if weight_path == "NO_LOAD":
                    weight_path = None
                elif weight_path is None:
                    weight_path = spec_path
            elif spec_path.suffix == ".yaml":
                config_yaml_path = spec_path
            else:
                raise NotImplementedError

            config_args = exp.parse_config_yaml(config_yaml_path, overwrite_config, override=False)
        else:
            config_args = overwrite_config

        if "model" not in config_args.keys():
            print("No model found.")
            return None

        basis_net_module = importlib.import_module("xcube.models." + config_args.model).Model

        if weight_path is not None:
            net_module = basis_net_module.load_from_checkpoint(weight_path, hparams=config_args)
        else:
            net_module = basis_net_module(config_args)

        return net_module

    @property
    def logger_type(self):
        logger = self.trainer.logger
        if logger is None:
            return 'none'
        elif 'Wandb' in type(logger).__name__:
            return 'wandb'
        else:
            return 'tb'

    def get_dataset_spec(self):
        raise NotImplementedError

    def get_collate_fn(self):
        raise NotImplementedError

    def get_hparams_metrics(self):
        raise NotImplementedError

    def train_val_step(self, *args, **kwargs):
        raise NotImplementedError

    def training_step(self, *args, **kwargs):
        try:
            return self.train_val_step(is_val=False, *args, **kwargs)
        except RuntimeError:
            # Compare to post-mortem, this would allow training to continue...
            exp.logger.warning(f"Training-step OOM. Skipping.")

            try:
                from xcube.data.base import DatasetSpec as DS
                exp.logger.warning(f"The problematic batch is: {args[0][DS.SHAPE_NAME]}")
            except:
                pass

            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            self.num_oom += 1.0
            self.log("num_oom", self.num_oom)
            return None

    def validation_step(self, *args, **kwargs):
        try:
            return self.train_val_step(is_val=True, *args, **kwargs)
        except RuntimeError:
            # Compare to post-mortem, this would allow training to continue...
            exp.logger.warning(f"Validation-step OOM. Skipping.")
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            self.num_oom += 1.0
            self.log("num_oom", self.num_oom)
            return None

    def on_before_overfit(self, batch):
        pass

    def tprint(self, *args):
        torch.set_printoptions(sci_mode=False, linewidth=200)
        if self.trainer is None or self.trainer.testing:
            print(*args)
        torch.set_printoptions(profile="default")

    def configure_optimizers(self):
        """
        Some comments the optimizers and schedulers.
            Usually just try SGD and Adam(W) will be fine, the latter of which does not depend too much on the
        learning rate. Other (RMSProp, AdaGrad, AdaDelta are bad).
            For schedulers, we use an approximation of exponential decay. Other strategies like polynomial or inverse
        square root are just alike, so we don't need to try them.
        """
        lr_config = self.hparams.learning_rate
        if self.hparams.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=lr_config['init'], momentum=0.9,
                                        weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'Adam':
            # AdamW corrects the bad weight dacay implementation in Adam.
            # AMSGrad also do some corrections to the original Adam.
            # The learning rate here is the maximum rate we can reach for each parameter.
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr_config['init'],
                                          weight_decay=self.hparams.weight_decay, amsgrad=True)
        else:
            raise NotImplementedError
        scheduler = LambdaLR(optimizer,
                             lr_lambda=functools.partial(
                                 lambda_lr_wrapper, lr_config=lr_config, batch_size=self.hparams.batch_size))
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def on_train_batch_start(self, batch: Any, batch_idx: int):
        exp.global_var_manager.set('skip_backward', False)

    def on_after_backward(self):
        # You can do gradient checking and clipping here. Directly on the models.
        # ... Haven't check what will happen for DDP though...
        if exp.global_var_manager.get('skip_backward'):
            exp.logger.info("Skip backward is enabled. This step won't affect the model.")
            for p in filter(lambda p: p.grad is not None, self.parameters()):
                p.grad.data.zero_()
            exp.global_var_manager.set('skip_backward', False)
            return

        # The following inspections can be added to logger by using --track_grad_norm 2.0
        grad_clip_val = self.hparams.get('grad_clip', 1000.)

        if grad_clip_val == "inspect":
            from pytorch_lightning.utilities.grads import grad_norm
            grad_dict = grad_norm(self, 'inf')      # Get the maximum absolute value.
            print(grad_dict)
            grad_clip_val = 1000.

        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=grad_clip_val)
        torch.nn.utils.clip_grad_value_(self.parameters(), clip_value=grad_clip_val)

        # If detect nan values, then this step is skipped
        has_nan_value_cnt = 0
        for p in filter(lambda p: p.grad is not None, self.parameters()):
            if torch.any(p.grad.data != p.grad.data):
                has_nan_value_cnt += 1
        if has_nan_value_cnt > 0:
            exp.logger.warning(f"{has_nan_value_cnt} parameters get nan-gradient -- this step will be skipped.")
            for p in filter(lambda p: p.grad is not None, self.parameters()):
                p.grad.data.zero_()

        # Also remove nan values if any.
        # has_nan_value_cnt = 0
        # for name, p in self.named_parameters():
        #     if p.grad is None:
        #         continue
        #     pdata = p.grad.data
        #     grad_is_nan = pdata != pdata
        #     if torch.any(grad_is_nan):
        #         has_nan_value_cnt += 1
        #         pdata[grad_is_nan] = 0.
        # if has_nan_value_cnt > 0:
        #     exp.logger.warning(f"{has_nan_value_cnt} parameters get nan-gradient -- they are set to 0.")

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        # print("Optimizer", optimizer_idx, "Step")
        pass

    def on_fit_start(self):
        if self.trainer.logger is None:
            return
        if self.logger_type == 'tb':
            writer = self.trainer.logger.experiment._get_file_writer()
            if writer is not None:
                hparams_metrics = {}
                for metric_name, should_add_best in self.get_hparams_metrics():
                    hparams_metrics[metric_name] = 0.0
                    hparams_metrics[f'best/{metric_name}'] = 0.0
                from pytorch_lightning.utilities.logger import (
                    _convert_params, _flatten_dict)
                hparams_dict = _convert_params(self.hparams)
                hparams_dict = _flatten_dict(hparams_dict)
                hparams_dict = self.trainer.logger._sanitize_params(hparams_dict)
                exp, ssi, sei = hparams(hparams_dict, hparams_metrics)
                writer.add_summary(exp)
                writer.add_summary(ssi)
                writer.add_summary(sei)
                # Note: only the name of the metrics, instead of their values (0.0) will be written.

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            logged_metrics = self.trainer.callback_metrics
            for metric_name, should_add_best in self.get_hparams_metrics():
                if should_add_best and metric_name in logged_metrics.keys():
                    self.best_metrics.append_loss({metric_name: logged_metrics[metric_name].item()})
                    # metric in hparams in tensorboard will automatically update according to added scalars.
                    #   so we do not need to manually update hyperparams.
                    self.log(f'best/{metric_name}', np.min(self.best_metrics.loss_dict[metric_name]))

    def log(self, name: str, value: Any, *args, **kwargs):
        """
            Override PL's default log function, to better support test-time or overfit-time no-logger logging
        (using various types including string).
        """
        # For overfitting, we leave the log to the log_dict monkey_patch.
        if isinstance(self.trainer, omegaconf.dictconfig.DictConfig):
            return
        if self.trainer.testing:
            if name == 'batch-idx':
                self.test_logged_values.append(OrderedDict([(name, value)]))
            else:
                if isinstance(value, torch.Tensor):
                    value = value.item()
                self.test_logged_values[-1][name] = value
        else:
            super().log(name=name, value=value, *args, **kwargs)

    def log_dict_prefix(
        self,
        prefix: str,
        dictionary: Mapping[str, Any],
        prog_bar: bool = False,
        logger: bool = True,
        on_step: Optional[bool] = None,
        on_epoch: Optional[bool] = None,
    ):
        """
        This overrides fixes if dict key is not a string...
        """
        dictionary = {
            prefix + "/" + str(k): v for k, v in dictionary.items()
        }
        self.log_dict(dictionary=dictionary,
                      prog_bar=prog_bar,
                      logger=logger, on_step=on_step, on_epoch=on_epoch)

    def log_image(self, name: str, img, step: Optional[int] = None):
        if self.trainer.logger is not None:
            if self.logger_type == 'tb':
                if img.shape[2] <= 4:
                    # WHC -> CWH
                    img = np.transpose(img, (2, 0, 1))
                self.trainer.logger.experiment.add_image(name, img, self.trainer.global_step)
            elif self.logger_type == 'wandb':
                # if img is a list
                if isinstance(img, list):
                    img = img
                else:
                    img = [img]

                self.trainer.logger.log_image(key=name, images=img)
                # self.trainer.logger.experiment.log({name: wandb.Image(img)})

    def log_video(self, name: str, video, fps=4):
        if self.trainer.logger is not None:
            if self.logger_type == 'wandb':
                self.trainer.logger.experiment.log({name: wandb.Video(video, fps=fps)})

    def log_plot(self, name: str, fig: matplotlib.figure.Figure, close: bool = True):
        img = exp.from_mplot(fig, close)
        self.log_image(name, img)
        if close:
            plt.close(fig)

    def test_log_data(self, data_dict: dict):
        # Output artifact data only when there is no focus.
        if self.record_folder is None or self.hparams.focus == "none":
            return
        self.record_data_cache.update(data_dict)

    def get_dataset_short_name(self):
        # Used for identify, e.g., camera paths.
        try:
            return self.trainer.test_dataloaders[0].dataset.get_short_name()
        except omegaconf.errors.ConfigAttributeError:
            return self.trainer.dataset_short_name

    def on_test_start(self):
        if self.hparams.get('record', None) is not None:
            from datetime import datetime
            test_set_name = self.trainer.test_dataloaders[0].dataset.get_name()
            if len(self.hparams.record) == 0:
                model_name = inspect.getfile(self.__class__).split('/')[-1].split('.py')[0]
                cur_time = datetime.now().strftime("%b%d-%X")
                save_folder_name = f"{cur_time}-{model_name}"
            else:
                save_folder_name = self.hparams.record[0]
            self.record_folder = Path("../test") / test_set_name / save_folder_name
            exp.mkdir_confirm(self.record_folder)
            exp.logger.info(f"Records will be saved at {self.record_folder}")
            with (self.record_folder / "hparams.yaml").open('w') as f:
                # yaml.dump(dict(self.hparams), f)
                OmegaConf.save(self.hparams, f)
            shutil.copy(f"models/{self.hparams.model.replace('.', '/')}.py", self.record_folder / "model.py")

        # Monkey-patch s.t. we can specify one batch to run on.
        if self.hparams.focus != "none":
            old_test_step = self.test_step
            def focus_test_step(batch, batch_idx):
                self.last_test_valid = False
                if self.hparams.focus != "all":
                    if self.hparams.focus[0] == "g":    # greater than
                        if batch_idx <= int(self.hparams.focus[1:]):
                            return
                    elif self.hparams.focus[0] == "l":  # less than
                        if batch_idx >= int(self.hparams.focus[1:]):
                            return
                    elif ',' in self.hparams.focus:
                        focus_ids = [int(t) for t in self.hparams.focus.split(',')]
                        if batch_idx not in focus_ids:
                            return
                    else:
                        if batch_idx != int(self.hparams.focus):
                            return
                old_test_step(batch, batch_idx)
                self.last_test_valid = True

            self.test_step = focus_test_step
        else:
            self.last_test_valid = True

    def print_test_logs(self):
        sample_log = self.test_logged_values[0]
        headers = list(sample_log.keys())
        headers.remove('batch-idx')
        print("Test logs:")
        for h in headers:
            header_values = [t[h] for t in self.test_logged_values if h in t]
            print(h, f"({len(header_values)})",
                  np.mean(header_values) if isinstance(header_values[0], float) else "NOT-NUMERIC")

    # Defines special data that requires separate folder to save
    TEST_SPECIAL_META = {
        dict: ["pth", lambda f, data: torch.save(data, f)],
        np.ndarray: ["npy", np.save]
    }

    def on_test_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int):
        self.log('batch-idx', batch_idx)

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        if not self.last_test_valid:
            return

        if self.record_folder is not None:
            # If has valid log
            if len(self.test_logged_values) > 0 and self.test_logged_values[-1]['batch-idx'] == batch_idx:
                cur_log = self.test_logged_values[-1]
                csv_path = self.record_folder / "result.csv"
                if len(self.record_headers) == 0:
                    self.record_headers = list(cur_log.keys())
                    self.record_headers.remove('batch-idx')
                    self.record_headers.insert(0, 'batch-idx')
                    with csv_path.open('w') as f:
                        f.write(",".join(self.record_headers) + "\n")
                with csv_path.open('a') as f:
                    # Cur log can be of other type:
                    value_list = []
                    for t in self.record_headers:
                        if t in cur_log.keys():
                            cur_log_t = cur_log[t]
                            value_list.append(str(cur_log_t))
                        else:
                            value_list.append("-")
                    f.write(",".join(value_list) + "\n")

            # Write output data if any.
            if len(self.record_data_cache) > 0:

                pkl_data = {k: v for k, v in self.record_data_cache.items()
                            if type(v) not in self.TEST_SPECIAL_META.keys()}
                special_data = {k: v for k, v in self.record_data_cache.items() if k not in pkl_data.keys()}

                for k, v in special_data.items():
                    (self.record_folder / k).mkdir(exist_ok=True, parents=True)
                    suffix, write_f = self.TEST_SPECIAL_META[type(v)]
                    write_f(str(self.record_folder / k / f"{batch_idx:06d}.{suffix}"), v)

                if len(pkl_data) > 0:
                    (self.record_folder / "test_log_data").mkdir(exist_ok=True, parents=True)

                    def convert_tensor(obj):
                        if isinstance(obj, tuple):
                            return tuple(map(convert_tensor, obj))
                        if isinstance(obj, list):
                            return list(map(convert_tensor, obj))
                        if isinstance(obj, dict):
                            return dict(map(convert_tensor, obj.items()))
                        if isinstance(obj, torch.Tensor):
                            return obj.detach().cpu().numpy()
                        return obj

                    try:
                        res = convert_tensor(pkl_data)
                    finally:
                        convert_tensor = None

                    with (self.record_folder / "test_log_data" / f"{batch_idx:06d}.pkl").open("wb") as f:
                        pickle.dump(res, f)

        self.record_data_cache = {}

    def dp_scatter(self, inputs, target_gpus, dim=0):
        """
        Define how dp scatter a batch of data. This only takes effect when using multiple GPUs.
        """
        from torch.nn.parallel.scatter_gather import scatter
        return scatter(inputs, target_gpus, dim)

    def _determine_batch_size(self):
        raise NotImplementedError

    def train_dataloader(self):
        # Note:
        import xcube.data as dataset
        train_set = dataset.build_dataset(
            self.hparams.train_dataset, self.get_dataset_spec(), self.hparams, self.hparams.train_kwargs)
        torch.manual_seed(0)
        return DataLoader(train_set, batch_size=self.hparams.batch_size // self.trainer.world_size, shuffle=True,
                          num_workers=self.hparams.train_val_num_workers, collate_fn=self.get_collate_fn())

    def val_dataloader(self):
        import xcube.data as dataset
        val_set = dataset.build_dataset(
            self.hparams.val_dataset, self.get_dataset_spec(), self.hparams, self.hparams.val_kwargs)
        return DataLoader(val_set, batch_size=self.hparams.batch_size // self.trainer.world_size, shuffle=False,
                          num_workers=self.hparams.train_val_num_workers, collate_fn=self.get_collate_fn())

    def test_dataloader(self):
        import xcube.data as dataset
        self.hparams.test_kwargs.resolution = self.hparams.resolution # ! use for testing when training on X^3 but testing on Y^3

        test_set = dataset.build_dataset(
            self.hparams.test_dataset, self.get_dataset_spec(), self.hparams, self.hparams.test_kwargs)
        if self.hparams.test_set_shuffle:
            torch.manual_seed(0)
        return DataLoader(test_set, batch_size=1, shuffle=self.hparams.test_set_shuffle, 
                          num_workers=0, collate_fn=self.get_collate_fn())

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

def lambda_lr_wrapper(it, lr_config, batch_size, accumulate_grad_batches=1):
    return max(
        lr_config['decay_mult'] ** (int(it * batch_size * accumulate_grad_batches / lr_config['decay_step'])),
        lr_config['clip'] / lr_config['init'])

class Model(BaseModel):
    def __init__(self, hparams):
        hparams = hparams_handler(hparams) # set up hparams automatically
        super().__init__(hparams)
        self.encoder = Encoder(self.hparams)           
        self.unet = eval(self.hparams.network.unet.target)(cut_ratio=self.hparams.cut_ratio, 
                                                           with_normal_branch=self.hparams.with_normal_branch,
                                                           with_semantic_branch=self.hparams.with_semantic_branch,
                                                           **self.hparams.network.unet.params)
        self.loss = Loss(self.hparams)
        
        # load pretrained weight
        if self.hparams.pretrained_weight is not None:
            logger.info(f"load pretrained weight from {self.hparams.pretrained_weight}")
            checkpoint = torch.load(self.hparams.pretrained_weight, map_location='cpu')
            missing_keys, unexpected_keys = self.load_state_dict(checkpoint['state_dict'], strict=False)
            logger.info(f"missing_keys: {missing_keys}")
            logger.info(f"unexpected_keys: {unexpected_keys}")

        # using for testing time
        self.reconstructor = None
    
    def build_hash_tree(self, input_xyz):
        if self.hparams.use_fvdb_loader:
            if isinstance(input_xyz, dict):
                return input_xyz
            return self.build_hash_tree_from_grid(input_xyz)
        
        return self.build_hash_tree_from_points(input_xyz)
    
    def build_hash_tree_from_points(self, input_xyz):
        if isinstance(input_xyz, torch.Tensor):
            input_xyz = fvdb.JaggedTensor(input_xyz)
        elif isinstance(input_xyz, fvdb.JaggedTensor):
            pass
        else:
            raise NotImplementedError
        
        hash_tree = {}
        for depth in range(self.hparams.tree_depth):
            if depth != 0 and not self.hparams.use_hash_tree:
                break
            voxel_size = [sv * 2 ** depth for sv in self.hparams.voxel_size]
            origins = [sv / 2. for sv in voxel_size]            
            hash_tree[depth] = fvdb.sparse_grid_from_nearest_voxels_to_points(input_xyz, 
                                                                              voxel_sizes=voxel_size, 
                                                                              origins=origins)
        return hash_tree
    
    def build_hash_tree_from_grid(self, input_grid):
        hash_tree = {}
        input_xyz = input_grid.grid_to_world(input_grid.ijk.float())
        
        for depth in range(self.hparams.tree_depth):
            if depth != 0 and not self.hparams.use_hash_tree:
                break            
            voxel_size = [sv * 2 ** depth for sv in self.hparams.voxel_size]
            origins = [sv / 2. for sv in voxel_size]
            
            if depth == 0:
                hash_tree[depth] = input_grid
            else:
                hash_tree[depth] = fvdb.sparse_grid_from_nearest_voxels_to_points(input_xyz, 
                                                                                  voxel_sizes=voxel_size, 
                                                                                  origins=origins)
        return hash_tree

    def forward(self, batch, out: dict):
        
        input_xyz = batch[DS.INPUT_PC]
        hash_tree = self.build_hash_tree(input_xyz)
        input_grid = hash_tree[0]
        batch.update({'input_grid': input_grid})

        if not self.hparams.use_hash_tree:
            hash_tree = None
                
        unet_feat = self.encoder(input_grid, batch)
        unet_feat = fvnn.VDBTensor(input_grid, input_grid.jagged_like(unet_feat))
        unet_res, unet_output, dist_features = self.unet(unet_feat, hash_tree)

        out.update({'tree': unet_res.structure_grid})
        out.update({
            'structure_features': unet_res.structure_features,
            'dist_features': dist_features,
        })
        out.update({'gt_grid': input_grid})
        out.update({'gt_tree': hash_tree})
        
        if self.hparams.with_normal_branch:
            out.update({
                'normal_features': unet_res.normal_features,
            })
        if self.hparams.with_semantic_branch:
            out.update({
                'semantic_features': unet_res.semantic_features,
            })
        if self.hparams.with_color_branch:
            out.update({
                'color_features': unet_res.color_features,
            })
        return out

    def on_validation_epoch_start(self):
        pass

    def train_val_step(self, batch, batch_idx, is_val):
        if batch_idx % 1 == 0:
            # Squeeze memory really hard :)
            # This is a trade-off between memory and speed
            gc.collect()
            torch.cuda.empty_cache()

        out = {'idx': batch_idx}
        out = self(batch, out)
        
        if out is None and not is_val:
            return None

        loss_dict, metric_dict, latent_dict = self.loss(batch, out, 
                                                        compute_metric=is_val, 
                                                        global_step=self.global_step,
                                                        current_epoch=self.current_epoch)

        if not is_val:
            self.log_dict_prefix('train_loss', loss_dict)
            self.log_dict_prefix('train_loss', latent_dict)
            if self.hparams.enable_anneal:
                self.log('anneal_kl_weight', self.loss.get_kl_weight(self.global_step))
        else:
            self.log_dict_prefix('val_metric', metric_dict)
            self.log_dict_prefix('val_loss', loss_dict)
            self.log_dict_prefix('val_loss', latent_dict)

        loss_sum = loss_dict.get_sum()
        self.log('val_loss' if is_val else 'train_loss/sum', loss_sum)
        self.log('val_step', self.global_step)

        return loss_sum

    def test_step(self, batch, batch_idx):        
        self.log('source', batch[DS.SHAPE_NAME][0])
        out = {'idx': batch_idx}
        out = self(batch, out)
        loss_dict, metric_dict, latent_dict = self.loss(batch, out, 
                                                        compute_metric=True, 
                                                        global_step=self.trainer.global_step,
                                                        current_epoch=self.current_epoch)
        self.log_dict(loss_dict)
        self.log_dict(metric_dict)
        self.log_dict(latent_dict)
        
    def get_dataset_spec(self):
        all_specs = [DS.SHAPE_NAME, DS.INPUT_PC,
                     DS.GT_DENSE_PC, DS.GT_GEOMETRY]
        if self.hparams.use_input_normal:
            all_specs.append(DS.TARGET_NORMAL)
            all_specs.append(DS.GT_DENSE_NORMAL)
        if self.hparams.use_input_semantic or self.hparams.with_semantic_branch:
            all_specs.append(DS.GT_SEMANTIC)
        if self.hparams.use_input_intensity:
            all_specs.append(DS.INPUT_INTENSITY)
        return all_specs

    def get_collate_fn(self):
        return list_collate

    def get_hparams_metrics(self):
        return [('val_loss', True)]

    def configure_optimizers(self):
        # overwrite this from base model to fix pretrained vae layer
        lr_config = self.hparams.learning_rate
        # parameters = list(self.parameters())
        parameters = list(self.encoder.parameters()) + list(self.unet.parameters())

        if self.hparams.optimizer == 'SGD':
            optimizer = torch.optim.SGD(parameters, lr=lr_config['init'], momentum=0.9,
                                        weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'Adam':
            # AdamW corrects the bad weight dacay implementation in Adam.
            # AMSGrad also do some corrections to the original Adam.
            # The learning rate here is the maximum rate we can reach for each parameter.
            optimizer = torch.optim.AdamW(parameters, lr=lr_config['init'],
                                          weight_decay=self.hparams.weight_decay, amsgrad=True)        
        else:
            raise NotImplementedError

        # build scheduler
        import functools

        from torch.optim.lr_scheduler import LambdaLR
        scheduler = LambdaLR(optimizer,
                             lr_lambda=functools.partial(
                                 lambda_lr_wrapper, lr_config=lr_config, batch_size=self.hparams.batch_size, accumulate_grad_batches=self.trainer.accumulate_grad_batches))

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    @torch.no_grad()
    def _encode(self, batch, use_mode=False):
        input_xyz = batch[DS.INPUT_PC]
        hash_tree = self.build_hash_tree(input_xyz)
        input_grid = hash_tree[0]
        batch.update({'input_grid': input_grid})

        if not self.hparams.use_hash_tree:
            hash_tree = None

        unet_feat = self.encoder(input_grid, batch)
        unet_feat = fvnn.VDBTensor(input_grid, input_grid.jagged_like(unet_feat))
        _, x, mu, log_sigma = self.unet.encode(unet_feat, hash_tree=hash_tree)
        if use_mode:
            sparse_feature = mu
        else:
            sparse_feature = reparametrize(mu, log_sigma)
        
        return fvnn.VDBTensor(x.grid, x.grid.jagged_like(sparse_feature))