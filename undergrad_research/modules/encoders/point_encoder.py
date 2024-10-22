# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch.nn as nn
import fvdb
import fvdb.nn as fvnn
from fvdb import JaggedTensor
from torch_scatter import scatter_max, scatter_mean

class ResnetBlockFC(nn.Module):
    """ Fully connected ResNet Block class. """
    def __init__(self, size_in: int, size_out: int = None, size_h: int = None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class PointNetEncoder(nn.Module):
    """ PointNet-based encoder network with ResNet blocks for each point.
        Number of input points are fixed.
    """

    def __init__(self,
                 dim: int,
                 c_dim: int = 32,
                 hidden_dim: int = 32,
                 n_blocks: int = 3):
        super().__init__()

        self.c_dim = c_dim
        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2 * hidden_dim, hidden_dim)
            for _ in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, c_dim)
        self.hidden_dim = hidden_dim

    def forward(self,
                pts_xyz: JaggedTensor,
                pts_feature: JaggedTensor,
                grid: fvdb.GridBatch):

        # Get voxel idx
        pts_xyz = grid.world_to_grid(pts_xyz)
        vid = grid.ijk_to_index(pts_xyz.round().int()).jdata

        # Map coordinates to local voxel
        pts_xyz = pts_xyz.jdata
        pts_xyz = (pts_xyz + 0.5) % 1

        pts_mask = vid != -1
        vid, pts_xyz = vid[pts_mask], pts_xyz[pts_mask]

        # Feature extraction
        if pts_feature is None:
            pts_feature = self.fc_pos(pts_xyz)
        else:
            pts_feature = pts_feature.jdata
            pts_feature = pts_feature[pts_mask]
            pts_feature = self.fc_pos(torch.cat([pts_xyz, pts_feature], dim=1))
        pts_feature = self.blocks[0](pts_feature)
        for block in self.blocks[1:]:
            pooled = scatter_max(pts_feature, vid, dim=0, dim_size=grid.total_voxels)[0]
            pooled = pooled[vid]
            pts_feature = torch.cat([pts_feature, pooled], dim=1)
            pts_feature = block(pts_feature)

        c = self.fc_c(pts_feature)
        c = scatter_mean(c, vid, dim=0, dim_size=grid.total_voxels)
        return grid.jagged_like(c)