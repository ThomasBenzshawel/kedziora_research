from modules.autoencoding.base_encoder import Encoder
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
from utils.Dataspec import DatasetSpec


# Third-party libraries - NumPy & Scientific
import numpy as np
from numpy.random import RandomState

# Third-party libraries - PyTorch
import torch
print (torch.__version__, "torch")
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
# from torch.utils.tensorboard.summary import hparams

# Third-party libraries - Visualization
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Third-party libraries - ML Tools
from omegaconf import DictConfig, ListConfig, OmegaConf
import omegaconf.errors

# Local imports
# from ext import common
import fvdb
import fvdb.nn as fvnn
from fvdb import JaggedTensor, GridBatch
from fvdb.nn import VDBTensor


from modules.autoencoding.sunet import StructPredictionNet
import collections


class UnetWrapper(nn.Module):
    def __init__(self, unet, hparams):
        super().__init__()
        self.encoder = Encoder(hparams)
        self.unet = unet
        self.hparams = hparams
        # Ensure cut_ratio has a default value if not provided
        if "cut_ratio" not in self.hparams:
            self.hparams["cut_ratio"] = 1.0  # Default value, adjust as needed
        

    def build_hash_tree_from_grid(self, input_grid):
        hash_tree = {}
        input_xyz = input_grid.grid_to_world(input_grid.ijk.float())
        
        for depth in range(self.hparams["tree_depth"]):
            if depth != 0 and not self.hparams["use_hash_tree"]:
                break            
            voxel_size = [sv * 2 ** depth for sv in self.hparams["voxel_size"]]
            origins = [sv / 2. for sv in voxel_size]
            
            if depth == 0:
                hash_tree[depth] = input_grid
            else:
                hash_tree[depth] = fvdb.gridbatch_from_nearest_voxels_to_points(input_xyz, 
                                                                                  voxel_sizes=voxel_size, 
                                                                                  origins=origins)
        return hash_tree

    def forward(self, batch, out: dict):
        input_xyz = batch[DatasetSpec.INPUT_PC]
        hash_tree = self.build_hash_tree_from_grid(input_xyz)
        input_grid = hash_tree[0]
        batch.update({'input_grid': input_grid})

        if not self.hparams["use_hash_tree"]:
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
        
        if self.hparams["with_normal_branch"]:
            out.update({
                'normal_features': unet_res.normal_features,
            })
        if self.hparams["with_semantic_branch"]:
            out.update({
                'semantic_features': unet_res.semantic_features,
            })
        if self.hparams["with_color_branch"]:
            out.update({
                'color_features': unet_res.color_features,
            })
        return out

    @torch.no_grad()
    def _encode(self, batch, use_mode=False):
        input_xyz = batch[DatasetSpec.INPUT_PC]
        hash_tree = self.build_hash_tree_from_grid(input_xyz)
        input_grid = hash_tree[0]
        batch.update({'input_grid': input_grid})

        if not self.hparams["use_hash_tree"]:
            hash_tree = None

        unet_feat = self.encoder(input_grid, batch)
        unet_feat = fvnn.VDBTensor(input_grid, input_grid.jagged_like(unet_feat))
        _, x, mu, log_sigma = self.unet.encode(unet_feat, hash_tree=hash_tree)
        if use_mode:
            sparse_feature = mu
        else:
            sparse_feature = reparametrize(mu, log_sigma)
        
        return fvnn.VDBTensor(x.grid, x.grid.jagged_like(sparse_feature))
    

    @staticmethod
    def load_from_checkpoint(checkpoint_path, industry_mapping):
        """Load the entire model from a checkpoint without needing separate autoencoder initialization."""
        # Load the entire checkpoint
        checkpoint = torch.load(checkpoint_path)


        u_net = StructPredictionNet(
            in_channels=checkpoint.get('in_channels'),    
            num_blocks=checkpoint.get('num_blocks'),
            f_maps=checkpoint.get('f_maps'),
            neck_dense_type=checkpoint.get('neck_dense_type'),
            neck_bound=checkpoint.get('neck_bound'),
            num_res_blocks=checkpoint.get('num_res_blocks'),
            use_residual=checkpoint.get('use_residual'),
            order=checkpoint.get('order'),
            is_add_dec=checkpoint.get('is_add_dec'),
            use_attention=checkpoint.get('use_attention'),
            use_checkpoint=checkpoint.get('use_checkpoint'),
            c_dim=checkpoint.get('c_dim')
        )


        unet_wrapper = UnetWrapper(u_net, {
            "tree_depth": checkpoint.get('tree_depth'),
            "voxel_size": checkpoint.get('voxel_size'),
            "use_hash_tree": checkpoint.get('use_hash_tree'),
            "use_input_normal": checkpoint.get('use_input_normal'),
            "use_input_semantic": checkpoint.get('use_input_semantic'),
            "use_input_color": checkpoint.get('use_input_color'),
            "use_input_intensity": checkpoint.get('use_input_intensity'),
            "c_dim": checkpoint.get('c_dim'),
            "with_normal_branch": checkpoint.get('with_normal_branch'),
            "with_semantic_branch": checkpoint.get('with_semantic_branch'),
            "with_color_branch": checkpoint.get('with_color_branch'),
        })

        # Load the state dict
        unet_wrapper .load_state_dict(checkpoint.get('model_state_dict', checkpoint))

        return unet_wrapper, u_net
    

    def get_config(self):
        return {
            "tree_depth": self.hparams["tree_depth"],
            "voxel_size": self.hparams["voxel_size"],
            "use_hash_tree": self.hparams["use_hash_tree"],
            "use_input_normal": self.hparams["use_input_normal"],
            "use_input_semantic": self.hparams["use_input_semantic"],
            "use_input_color": self.hparams["use_input_color"],
            "use_input_intensity": self.hparams["use_input_intensity"],
            "c_dim": self.hparams["c_dim"],
            "with_normal_branch": self.hparams["with_normal_branch"],
            "with_semantic_branch": self.hparams["with_semantic_branch"],
            "with_color_branch": self.hparams["with_color_branch"],
            "cut_ratio": self.hparams.get("cut_ratio", 1.0),  # Added cut_ratio with default
        }

    @staticmethod
    def load_from_checkpoint(checkpoint_path, industry_mapping):
        """Load the entire model from a checkpoint without needing separate autoencoder initialization."""
        # Load the entire checkpoint
        checkpoint = torch.load(checkpoint_path)

        u_net = StructPredictionNet(
            in_channels=checkpoint.get('in_channels'),    
            num_blocks=checkpoint.get('num_blocks'),
            f_maps=checkpoint.get('f_maps'),
            neck_dense_type=checkpoint.get('neck_dense_type'),
            neck_bound=checkpoint.get('neck_bound'),
            num_res_blocks=checkpoint.get('num_res_blocks'),
            use_residual=checkpoint.get('use_residual'),
            order=checkpoint.get('order'),
            is_add_dec=checkpoint.get('is_add_dec'),
            use_attention=checkpoint.get('use_attention'),
            use_checkpoint=checkpoint.get('use_checkpoint'),
            c_dim=checkpoint.get('c_dim')
        )

        unet_wrapper = UnetWrapper(u_net, {
            "tree_depth": checkpoint.get('tree_depth'),
            "voxel_size": checkpoint.get('voxel_size'),
            "use_hash_tree": checkpoint.get('use_hash_tree'),
            "use_input_normal": checkpoint.get('use_input_normal'),
            "use_input_semantic": checkpoint.get('use_input_semantic'),
            "use_input_color": checkpoint.get('use_input_color'),
            "use_input_intensity": checkpoint.get('use_input_intensity'),
            "c_dim": checkpoint.get('c_dim'),
            "with_normal_branch": checkpoint.get('with_normal_branch'),
            "with_semantic_branch": checkpoint.get('with_semantic_branch'),
            "with_color_branch": checkpoint.get('with_color_branch'),
            "cut_ratio": checkpoint.get('cut_ratio', 1.0),  # Added cut_ratio with default
        })

        # Load the state dict
        unet_wrapper.load_state_dict(checkpoint.get('model_state_dict', checkpoint))

        return unet_wrapper, u_net

    def save_checkpoint(self, filepath):
        """Save the model weights and configuration to a checkpoint file."""
        checkpoint = {
            # Configuration parameters
            **self.get_config(),  # Unpack all config parameters from get_config
            
            # Add U-Net specific parameters that aren't in get_config
            'in_channels': self.unet.in_channels,
            'num_blocks': self.unet.num_blocks,
            'f_maps': self.unet.f_maps,
            'neck_dense_type': self.unet.neck_dense_type,
            'neck_bound': self.unet.neck_bound,
            'num_res_blocks': self.unet.num_res_blocks,
            'use_residual': self.unet.use_residual,
            'order': self.unet.order,
            'is_add_dec': self.unet.is_add_dec,
            'use_attention': self.unet.use_attention,
            'use_checkpoint': self.unet.use_checkpoint,
            
            # Model weights
            'model_state_dict': self.state_dict(),
        }
        
        torch.save(checkpoint, filepath)