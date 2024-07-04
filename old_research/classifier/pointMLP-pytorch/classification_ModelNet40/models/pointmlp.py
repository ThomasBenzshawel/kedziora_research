import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch import einsum
# from einops import rearrange, repeat

from pointnet2_ops import pointnet2_utils

device = "cpu"


def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class LocalGrouper(nn.Module):
    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize="center", **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel=3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1,1,1,channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]

        # fps_idx = torch.multinomial(torch.linspace(0, N - 1, steps=N).repeat(B, 1).to(xyz.device), num_samples=self.groups, replacement=False).long()
        # fps_idx = farthest_point_sample(xyz, self.groups).long()
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.groups).long()  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        new_points = index_points(points, fps_idx)  # [B, npoint, d]

        idx = knn_point(self.kneighbors, xyz, new_xyz)
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
        grouped_points = index_points(points, idx)  # [B, npoint, k, d]
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz],dim=-1)  # [B, npoint, k, d+3]
        if self.normalize is not None:
            if self.normalize =="center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize =="anchor":
                mean = torch.cat([new_points, new_xyz],dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]
            std = torch.std((grouped_points-mean).reshape(B,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
            grouped_points = (grouped_points-mean)/(std + 1e-5)
            grouped_points = self.affine_alpha*grouped_points + self.affine_beta

        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        return new_xyz, new_points


class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class PreExtraction(nn.Module):
    def __init__(self, channels, out_channels,  blocks=1, groups=1, res_expansion=1, bias=True,
                 activation='relu', use_xyz=True):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PreExtraction, self).__init__()
        in_channels = 3+2*channels if use_xyz else 2*channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion,
                                bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.operation(x)  # [b, d, k]
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu'):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):  # [b, d, g]
        return self.operation(x)


class Model(nn.Module):
    def __init__(self, points=1024, class_num=40, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="center",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[2, 2, 2, 2], **kwargs):
        super(Model, self).__init__()
        self.stages = len(pre_blocks)
        self.class_num = class_num
        self.points = points
        self.embedding = ConvBNReLU1D(3, embed_dim, bias=bias, activation=activation)
        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        anchor_points = self.points
        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            # append local_grouper_list
            local_grouper = LocalGrouper(last_channel, anchor_points, kneighbor, use_xyz, normalize)  # [b,g,k,d]
            self.local_grouper_list.append(local_grouper)
            # append pre_block_list
            pre_block_module = PreExtraction(last_channel, out_channel, pre_block_num, groups=groups,
                                             res_expansion=res_expansion,
                                             bias=bias, activation=activation, use_xyz=use_xyz)
            self.pre_blocks_list.append(pre_block_module)
            # append pos_block_list
            pos_block_module = PosExtraction(out_channel, pos_block_num, groups=groups,
                                             res_expansion=res_expansion, bias=bias, activation=activation)
            self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel

        self.act = get_activation(activation)
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, 512),
            nn.BatchNorm1d(512),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(256, self.class_num)
        )

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.embedding(x)  # B,D,N
        for i in range(self.stages):
            # Give xyz[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, g, k, d]
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))  # [b,g,3]  [b,g,k,d]
            x = self.pre_blocks_list[i](x)  # [b,d,g]
            x = self.pos_blocks_list[i](x)  # [b,d,g]

        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
        x = self.classifier(x)
        return x




def pointMLP(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=64, groups=1, res_expansion=1.0,
                   activation="relu", bias=False, use_xyz=False, normalize="anchor",
                   dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                   k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], **kwargs)


def pointMLPElite(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=32, groups=1, res_expansion=0.25,
                   activation="relu", bias=False, use_xyz=False, normalize="anchor",
                   dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                   k_neighbors=[24,24,24,24], reducers=[2, 2, 2, 2], **kwargs)

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels//4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels//4, 1, bias=False)
        # self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.conv = nn.Conv1d(3, channels, 1)

    # def execute(self, x):
    def forward(self, x, xyz,t):
        pos = self.conv(xyz)
        x = x + pos
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv(x)# b, c, n        
        x_v = self.v_conv(x) +pos
        x_v = x_v *t
        energy = torch.bmm(x_q, x_k) # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True)) 
        x_r = torch.bmm(x_v,attention) # b, c, n 
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    distance = pairwise_distance.topk(k=k, dim=-1)[0]
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return distance,idx

def get_graph_feature(xyz,x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)          #x.size()  B*C*N
    concat = torch.empty(batch_size,4,num_points,k)
    coords = xyz.permute([0, 2, 1])
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            dis, idx = knn(xyz, k=k)   # idx (batch_size, num_points, k)
        else:
            dis, idx = knn(x[:, 6:,], k=k)
            
    B, N, K = idx.size()
    # idx(B, N, K), coords(B, N, 3)
    # neighbors[b, i, n, k] = coords[b, idx[b, n, k], i] = extended_coords[b, i, extended_idx[b, i, n, k], k]
    extended_idx = idx.unsqueeze(1).expand(B, 3, N, K)
    extended_coords = coords.transpose(-2,-1).unsqueeze(-1).expand(B, 3, N, K)
    neighbors = torch.gather(extended_coords, 2, extended_idx) # shape (B, 3, N, K)     
    pos =  torch.cat((
            extended_coords - neighbors,
            dis.unsqueeze(-3)
    ), dim=-3).to(device) # B 10 N K
    pos = pos.permute(0,2,3,1)
    # concat = torch.cat((
    #         extended_coords,
    #         neighbors,
    #         extended_coords - neighbors,
    #         dis.unsqueeze(-3)
    # ), dim=-3).to(device) # B 10 N K

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)              #B*N*k
 
    _, num_dims, _ = x.size()


    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]  #[B*N*k,C]

    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    x = feature-x
    pos = torch.cat((pos,x), dim=3).permute(0, 3, 1, 2).contiguous()
    feature = x.permute(0, 3, 1, 2).contiguous()
    return pos, feature      # (batch_size, 2*num_dims, num_points, k)


class IBT_cls(nn.Module):
    # def __init__(self, args ,d_points, d_model, dropout=0,alpha=0.2):
    def __init__(self, dropout=0,alpha=0.2):
        super(IBT_cls, self).__init__()
        self.k = 40 
        self.dropout = dropout
        # self.transform_net = Transform_Net(args)
        # self.pointtrans = PointTransformerLayer(d_points, d_model, self.k)
        self.trans0 = SA_Layer(128)
        self.trans1 = SA_Layer(128)
        self.trans2 = SA_Layer(128)
        self.trans3 = SA_Layer(1024)
        
        self.bn_p1 = nn.BatchNorm1d(32)
        self.bn_p2 = nn.BatchNorm1d(64)
        self.bn_p3 = nn.BatchNorm1d(128)
        self.bn_c0 = nn.BatchNorm2d(64)
        self.bn_c1 = nn.BatchNorm2d(64)
        self.bn_c2 = nn.BatchNorm2d(64)
        self.bn_c3 = nn.BatchNorm2d(64)

        self.bn = nn.BatchNorm1d(64)
        self.bn0 = nn.BatchNorm2d(64*2)
        self.bn1 = nn.BatchNorm2d(64*2)
        self.bn2 = nn.BatchNorm2d(64*2)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(128)
        self.bn6 = nn.BatchNorm1d(128)
        self.bn7 = nn.BatchNorm1d(128)#emb_dims)
        self.bn10 = nn.BatchNorm1d(64*2)
        self.bn11 = nn.BatchNorm1d(64)
        self.bn12 = nn.BatchNorm1d(64)

        self.point1 = nn.Sequential(nn.Conv1d(3, 32, kernel_size=1, bias=False),self.bn_p1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.point2 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=1, bias=False),self.bn_p2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.point3 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1, bias=False),self.bn_p3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.mlp0 = nn.Sequential(nn.Conv2d(4+128, 64, kernel_size=1, bias=False),self.bn_c0,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.mlp1 = nn.Sequential(nn.Conv2d(4+128, 64, kernel_size=1, bias=False),self.bn_c1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.mlp2 = nn.Sequential(nn.Conv2d(4+128, 64, kernel_size=1, bias=False),self.bn_c2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.mlp3 = nn.Sequential(nn.Conv2d(4, 64, kernel_size=1, bias=False),self.bn_c3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv = nn.Sequential(nn.Conv1d(3, 64, kernel_size=1, bias=False),self.bn,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv0 = nn.Sequential(nn.Conv2d(64*3, 64*2, kernel_size=1, bias=False),self.bn0,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv1 = nn.Sequential(nn.Conv2d(64*3, 64*2, kernel_size=1, bias=False),self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        # self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),self.bn2,
        #                            nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*3, 64*2, kernel_size=1, bias=False),self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv1d(128*2, 128, kernel_size=1, bias=False),self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(128*2, 128, kernel_size=1, bias=False),self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(128*2, 128, kernel_size=1, bias=False),self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        #self.conv7 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),self.bn7,
        self.conv7 = nn.Sequential(nn.Conv1d(512, 128, kernel_size=1, bias=False),self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv10 = nn.Sequential(nn.Conv1d(64, 64*2, kernel_size=1, bias=False),self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=False),self.bn11,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv12 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=False),self.bn12,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(128, 512, bias=False)
        self.bn8 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=self.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn9 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=self.dropout)
        self.linear3 = nn.Linear(256, 15)

        self.score_fn0 = nn.Sequential(
            nn.Linear(128, 128, bias=False),
            nn.Softmax(dim=-2)
        )
        self.score_fn1 = nn.Sequential(
            nn.Linear(128, 128, bias=False),
            nn.Softmax(dim=-2)
        )
        self.score_fn2 = nn.Sequential(
            nn.Linear(128, 128, bias=False),
            nn.Softmax(dim=-2)
        )
        self.score_fn3 = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.Softmax(dim=-2)
        )



    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)
        xyz = x[:,:3,:]     #B,C,N

        x00 = self.point1(x)
        x_p = self.point2(x00)
        x_p = self.point3(x_p)
    
        # concat,x0 = get_graph_feature(xyz,x, k=self.k)     # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        # t = self.transform_net(x0)              # (batch_size, 3, 3)
        # x = x.transpose(2, 1)                   # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        # x = torch.bmm(x, t)                     # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        # x = x.transpose(2, 1)                   # (batch_size, num_points, 3) -> (batch_size, 3, num_points)
        # x = self.conv(x)

        x00 = x_p.max(dim=-1, keepdim=False)[0]
        x00 = x00.unsqueeze(-1)
        x00 =x00.repeat(1,1,num_points)

        pos, x = get_graph_feature(xyz,x_p, k=self.k)   # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        pos = self.mlp0(pos)
        x = torch.cat([pos, x], dim=1)
        x = self.conv0(x)                       # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x0 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        #x0 = x.sum(dim=-1, keepdim=False) 
        scores = self.score_fn0(x.permute(0,2,3,1)).permute(0,3,1,2)
        features = torch.sum(scores * x, dim=-1, keepdim=True).squeeze(-1)
        # x0=x0 + features
        x0 = torch.cat((x0,features),dim=1)
        x0 =self.conv4(x0)
        t = torch.sigmoid(x0)
        x_t = self.trans0(x_p,xyz,t)
        x0=x0+x_t
        # x0 = torch.cat((x0,x_t),dim=1)
        # x0 = self.conv10(x0)


        #---------------------------------------------------------------------------------------------------------------------------

        pos, x = get_graph_feature(xyz,x0, k=self.k)   # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        pos = self.mlp1(pos)
        x = torch.cat([pos, x], dim=1)
        x = self.conv1(x)                       # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        # x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        #x1 = x.sum(dim=-1, keepdim=False)
        scores = self.score_fn1(x.permute(0,2,3,1)).permute(0,3,1,2)
        features = torch.sum(scores * x, dim=-1, keepdim=True).squeeze(-1)
        # x1=x1 + features
        x1 = torch.cat((x1,features),dim=1)
        x1 =self.conv5(x1)
        t = torch.sigmoid(x1)
        x_t = self.trans1(x0,xyz,t)
        x1 = x1+x_t
        # x1 = torch.cat((x1,x_t),dim=1)
        # x1 = self.conv11(x1)

        #---------------------------------------------------------------------------------------------------------------------------

        #---------------------------------------------------------------------------------------------------------------------------
        pos, x = get_graph_feature(xyz,x1, k=self.k)   # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        pos = self.mlp2(pos)
        x = torch.cat([pos, x], dim=1)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        # x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        #x2 = x.sum(dim=-1, keepdim=False)
        scores = self.score_fn2(x.permute(0,2,3,1)).permute(0,3,1,2)
        features = torch.sum(scores * x, dim=-1, keepdim=True).squeeze(-1)
        # x2=x2 + features
        x2 = torch.cat((x2,features),dim=1)
        x2 =self.conv6(x2)
        t = torch.sigmoid(x2)
        x_t = self.trans2(x1,xyz,t)
        x2 = x2 + x_t
        # x2 = torch.cat((x2,x_t),dim=1)
        # x2 = self.conv12(x2)

        #---------------------------------------------------------------------------------------------------------------------------

        #---------------------------------------------------------------------------------------------------------------------------

        # pos, x = get_graph_feature(xyz,x2, k=self.k)   # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        # pos = self.mlp3(pos)
        # x = torch.cat([pos, x], dim=1)
        # x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        # x = self.conv6(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        # x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        # #x3 = x.sum(dim=-1, keepdim=False)
        # scores = self.score_fn3(x.permute(0,2,3,1)).permute(0,3,1,2)
        # features = torch.sum(scores * x, dim=-1, keepdim=True).squeeze()
        # x3=x3 + features


        x = torch.cat((x00,x0,x1, x2), dim=1)      # (batch_size, 64*3, num_points)
        #x = torch.cat((x0,x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)
        x = self.conv7(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)

        x = F.leaky_relu(self.bn8(self.linear1(x)), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn9(self.linear2(x)), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)                                             # (batch_size, 256) -> (batch_size, output_channels)
        
        return x


if __name__ == '__main__':
    data = torch.rand(2, 3, 1024)
    print("===> testing pointMLP ...")
    model = pointMLP()
    #model = IBT_cls()
    out = model(data)
    print(out.shape)

