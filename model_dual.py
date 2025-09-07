import numpy as np
import os
import torch
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torch import nn
import timm
import config as c
from freia_funcs import *
# from cs_util.freia_funcs import *
from fightingcv_attention.attention.SEAttention import SEAttention


MODEL_DIR = './models_cfm'

def get_nf(input_dim=c.n_feat, channels_hidden=c.channels_hidden_teacher):
    nodes = list()
    if c.pos_enc:
        nodes.append(InputNode(c.pos_enc_dim, name='input'))

    nodes.append(InputNode(input_dim, name='input'))
    for k in range(c.n_coupling_blocks):
        nodes.append(Node([nodes[-1].out0], permute_layer, {'seed': k}, name=F'permute_{k}'))
        if c.pos_enc:
            nodes.append(Node([nodes[-1].out0, nodes[0].out0], glow_coupling_layer_cond,
                              {'clamp': c.clamp,
                               'F_class': F_conv_attention,
                               'cond_dim': c.pos_enc_dim,
                               'F_args': {'channels_hidden': channels_hidden,
                                          'kernel_size': c.kernel_sizes[k]}},
                              name=F'conv_{k}'))
        else:
            nodes.append(Node([nodes[-1].out0], glow_coupling_layer_cond,
                              {'clamp': c.clamp,
                               'F_class': F_conv_attention,
                               'F_args': {'channels_hidden': channels_hidden,
                                          'kernel_size': c.kernel_sizes[k]}},
                              name=F'conv_{k}'))

    nodes.append(OutputNode([nodes[-1].out0], name='output'))
    nf = ReversibleGraphNet(nodes, n_jac=1)
    return nf


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.feature_extractor = EfficientNet.from_pretrained('efficientnet-b5')
        self.layer_idx = [19, 26, 35]

    def forward(self, x):
        features = list()
        x = self.feature_extractor._swish(self.feature_extractor._bn0(self.feature_extractor._conv_stem(x)))
        # Blocks
        for idx, block in enumerate(self.feature_extractor._blocks):
            drop_connect_rate = self.feature_extractor._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.feature_extractor._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx in self.layer_idx:
                features.append(x)

        feature1 = torch.cat([features[0],features[1]],dim=1)
        feature1 = F.interpolate(feature1,(24,24),mode="bilinear")
        feature = torch.cat([feature1,features[2]],dim=1)
        return feature

def positionalencoding2d(D, H, W):
    """
    taken from https://github.com/gudovskiy/cflow-ad
    :param D: dimension of the model
    :param H: H of the positions
    :param W: W of the positions
    :return: DxHxW position matrix
    """
    if D % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(D))
    P = torch.zeros(D, H, W)
    # Each dimension use half of D
    D = D // 2
    div_term = torch.exp(torch.arange(0.0, D, 2) * -(np.log(1e4) / D))
    pos_w = torch.arange(0.0, W).unsqueeze(1)
    pos_h = torch.arange(0.0, H).unsqueeze(1)
    P[0:D:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[1:D:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[D::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    P[D + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    return P.to(c.device)[None]


class FeatureProjectionConv(nn.Module):
    def __init__(self, inchannel=1152, outchannel=608):
        super().__init__()
        self.conv1 = nn.Conv2d(inchannel, (inchannel+outchannel)//2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d((inchannel+outchannel)//2, (inchannel+outchannel)//4, kernel_size=3, padding=1)

        self.conv3 = nn.ConvTranspose2d((inchannel+outchannel)//4, 128, kernel_size=2, stride=2)
        self.se = SEAttention(channel=128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.upconv1 = nn.ConvTranspose2d(256, 512, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(512, 608, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(608, 608, kernel_size=3, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        out = self.relu(self.conv4(out))
        out = self.upconv2(self.upconv1(out))
        out = self.relu(self.conv5(out))

        return out


class Model(nn.Module):
    def __init__(self, nf=True, n_blocks=c.n_coupling_blocks, channels_hidden=c.channels_hidden_teacher, depth = False):
        super(Model, self).__init__()

        if not c.pre_extracted:
            self.feature_extractor = FeatureExtractor(layer_idx=c.extract_layer)

        if nf:
            self.net = get_nf(input_dim=c.n_feat)

        if c.pos_enc:
            self.pos_enc = positionalencoding2d(c.pos_enc_dim, c.map_len, c.map_len)

        self.unshuffle = nn.PixelUnshuffle(c.depth_downscale)

    def forward(self, x, depth, ano, depth_feature, flag = False):
        if not c.pre_extracted and c.mode != 'depth':
            with torch.no_grad():
                f = self.feature_extractor(x)
        else:
            f = x

        if flag:
            inp = depth_feature
        else:
            inp = torch.cat([f, self.unshuffle(depth) + self.unshuffle(ano)], dim=1)

        if c.pos_enc:
            cond = self.pos_enc.tile(inp.shape[0], 1, 1, 1)
            z = self.net([cond, inp])
        else:
            z = self.net(inp)
        jac = self.net.jacobian(run_forward=False)[0]
        return z, jac


def save_weights(model, suffix):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model.to('cpu')
    torch.save(model.net.state_dict(), join(MODEL_DIR, f'{c.modelname}_{c.class_name}_{suffix}.pth'))
    print('model saved')
    model.to(c.device)


def load_weights(model, suffix):
    model.net.load_state_dict(torch.load(os.path.join(MODEL_DIR, f'{c.modelname}_{c.class_name}_{suffix}.pth')))
    model.eval()
    model.to(c.device)
    return model


def save_weights_cfm(model, suffix):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model.to('cpu')
    torch.save(model.state_dict(), join(MODEL_DIR, f'{c.modelname}_{c.class_name}_{suffix}.pth'))
    print('model saved')
    model.to(c.device)


def load_weights_cfm(model, suffix):
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f'{c.modelname}_{c.class_name}_{suffix}.pth')))
    model.eval()
    model.to(c.device)
    return model
