import os
import numpy as np
import torch
import torch.nn as nn
import timm
from timm.models.layers import DropPath, trunc_normal_
from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
import config as c

from os.path import join
import math

def fps(data, number):
    '''
        data : B N 3
        number : int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return fps_data, fps_idx


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center, center_idx = fps(xyz.contiguous(), self.num_group)  # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        ori_idx = idx
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.reshape(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.reshape(batch_size, self.num_group, self.group_size, 3).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center, ori_idx, center_idx


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        #  3xBxheadsxNxdim
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # 1xBxheadxNxdim
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        feature_list = []
        # 只取指定特征的维度
        fetch_idx = [3,7,11]
        # TODO
        # fetch_idx = [3,7]
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if i in fetch_idx:
                # return x
                feature_list.append(x)
        return feature_list


class PointTransformer(nn.Module):
    def __init__(self, group_size=128, num_group=1024, encoder_dims=384):
        super().__init__()

        self.trans_dim = 384
        self.depth = 12
        self.drop_path_rate = 0.1
        self.num_heads = 6

        self.group_size = group_size
        self.num_group = num_group
        # grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # define the encoder
        self.encoder_dims = encoder_dims
        if self.encoder_dims != self.trans_dim:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
            self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        # bridge encoder and transformer

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)


    def load_model_from_pb_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            del base_ckpt[k]

        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print('missing_keys')
            print(
                incompatible.missing_keys
            )
        if incompatible.unexpected_keys:
            print('unexpected_keys')
            print(
                incompatible.unexpected_keys

            )

        print(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}')

    def forward(self, pts):
        if self.encoder_dims != self.trans_dim:
            B, C, N = pts.shape
            pts = pts.transpose(-1, -2)  # B N 3
            # divide the point clo  ud in the same form. This is important
            neighborhood, center, ori_idx, center_idx = self.group_divider(pts)
            # # generate mask
            # bool_masked_pos = self._mask_center(center, no_mask = False) # B G
            # encoder the input cloud blocks
            group_input_tokens = self.encoder(neighborhood)  # B G N
            group_input_tokens = self.reduce_dim(group_input_tokens)
            # prepare cls
            cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
            cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
            # add pos embedding
            pos = self.pos_embed(center)
            # final input
            x = torch.cat((cls_tokens, group_input_tokens), dim=1)
            pos = torch.cat((cls_pos, pos), dim=1)
            # transformer
            feature_list = self.blocks(x, pos)
            feature_list = [self.norm(x)[:, 1:].transpose(-1, -2).contiguous() for x in feature_list]
            x = torch.cat((feature_list[0], feature_list[1], feature_list[2]), dim=1)  # 1152
            return x, center, ori_idx, center_idx
        else:
            B, C, N = pts.shape
            pts = pts.transpose(-1, -2)  # B N 3
            # divide the point clo  ud in the same form. This is important
            neighborhood, center, ori_idx, center_idx = self.group_divider(pts)

            group_input_tokens = self.encoder(neighborhood)  # B G N

            pos = self.pos_embed(center)
            # final input
            x = group_input_tokens
            # transformer
            feature_list = self.blocks(x, pos)
            feature_list = [self.norm(x).transpose(-1, -2).contiguous() for x in feature_list]
            x = torch.cat((feature_list[0], feature_list[1], feature_list[2]), dim=1)
            return x, center, ori_idx, center_idx

def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
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
    # dist --> BxNxM
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


def interpolating_points(xyz1, xyz2, points2):
    """
    Input:
        xyz1: input points position data, [B, C, N]
        xyz2: sampled input points position data, [B, C, S]
        points2: input points data, [B, D, S]
    Return:
        new_points: upsampled points data, [B, D', N]
    """
    # BxNxC
    xyz1 = xyz1.permute(0, 2, 1)
    # BxSxC
    xyz2 = xyz2.permute(0, 2, 1)

    # BxSxD
    points2 = points2.permute(0, 2, 1)
    B, N, C = xyz1.shape
    _, S, _ = xyz2.shape

    if S == 1:
        # only one sample data, repeat it N times
        interpolated_points = points2.repeat(1, N, 1)
    else:
        dists = square_distance(xyz1, xyz2)
        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

        # use inverse distance weight
        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        interpolated_points = interpolated_points.permute(0, 2, 1)
    return interpolated_points


def organized_pc_to_unorganized_pc(organized_pc):
    # WxHxC ==> WH x C
    return organized_pc.reshape(organized_pc.shape[0] * organized_pc.shape[1], organized_pc.shape[2])

def unorganized_pc_to_organized_pc(unorganized_pc):
    # WHxC ==> W x H x C
    n = int(math.sqrt(unorganized_pc.shape[0]))
    return unorganized_pc.reshape(n, n, unorganized_pc.shape[1])



def point_extractor(pcs):
    model = PointTransformer(group_size=128, num_group=1024)
    model.load_model_from_ckpt("model_weight")
    model.to(c.device)
    resize = torch.nn.AdaptiveAvgPool2d((24, 24))
    average = torch.nn.AvgPool2d(kernel_size=3, stride=1)

    point_features = list()
    with torch.no_grad():
        for i in range(pcs.shape[0]):
            organized_pc = pcs[i]
            organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
            unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
            nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]

            unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(
                0, 2, 1).to(c.device)
            xyz_feature_maps, center, neighbor_idx, center_idx = model(unorganized_pc_no_zeros.contiguous())

            interpolated_pc = interpolating_points(unorganized_pc_no_zeros.contiguous(),
                                                   center.permute(0, 2, 1), xyz_feature_maps)

            xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], 224 * 224),
                                         dtype=xyz_feature_maps.dtype, device=c.device)
            xyz_patch_full[..., nonzero_indices] = interpolated_pc
            xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], 224, 224)
            xyz_patch_full_resized = resize(average(xyz_patch_full_2d))

            point_features.append(xyz_patch_full_resized)
        point_features = torch.concatenate(point_features, dim=0)

        return point_features