import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

class SEAttention(nn.Module):
    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)

        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output



class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        # conv则是实际进行的卷积操作，注意这里步长设置为卷积核大小，因为与该卷积核进行卷积操作的特征图是由输出特征图中每个点扩展为其对应卷积核那么多个点后生成的。
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        # p_conv是生成offsets所使用的卷积，输出通道数为卷积核尺寸的平方的2倍，代表对应卷积核每个位置横纵坐标都有偏移量。
        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation  # modulation是可选参数,若设置为True,那么在进行卷积操作时,对应卷积核的每个位置都会分配一个权重。
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        # 由于卷积核中心点位置是其尺寸的一半，于是中心点向左（上）方向移动尺寸的一半就得到起始点，向右（下）方向移动另一半就得到终止点
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        # p0_y、p0_x就是输出特征图每点映射到输入特征图上的纵、横坐标值。
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride))

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    # 输出特征图上每点（对应卷积核中心）加上其对应卷积核每个位置的相对（横、纵）坐标后再加上自学习的（横、纵坐标）偏移量。
    # p0就是将输出特征图每点对应到卷积核中心，然后映射到输入特征图中的位置；
    # pn则是p0对应卷积核每个位置的相对坐标；
    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        # 计算双线性插值点的4邻域点对应的权重
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset

class DeformableChannelAttention(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=None):
        super().__init__()
        self.deformable_conv = DeformConv2d(in_channel, out_channel, kernel_size, padding, stride, bias)
        self.se = SEAttention(out_channel)

    def forward(self, x):
        ori_x = x.clone()
        x = self.deformable_conv(x)
        attention_map = self.se(x)
        x = x * attention_map
        # x = ori_x + x
        return x

class DeformableSpatialAttention(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=None):
        super().__init__()
        self.deformable_conv = DeformConv2d(in_channel, out_channel, kernel_size, padding, stride, bias)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        ori_x = x.clone()
        x = self.deformable_conv(x)
        attention_map = self.sa(x)
        x = x * attention_map
        return x

class DeformRegularChannelAttention(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=None):
        super().__init__()
        self.channel1 = out_channel - out_channel // 2
        self.channel2 = out_channel // 2
        self.deformable_conv = DeformConv2d(in_channel, self.channel1, kernel_size, padding, stride, bias)
        self.conv = nn.Conv2d(in_channel, self.channel2, kernel_size=kernel_size, padding=padding, stride=stride)
        self.se = SEAttention(out_channel)

    def forward(self, x):
        deform_x = self.deformable_conv(x)
        conv_x = self.conv(x)
        out = torch.cat([deform_x, conv_x], dim=1)
        out = self.se(out)
        return out


class MultiScaleAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleAttention, self).__init__()

        # 定义不同尺度的卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)

        self.conv4 = nn.Conv2d(out_channels*3, out_channels, kernel_size=1)

        # 线性层用于动态调整
        self.fc = nn.Linear(out_channels * 3, out_channels)


    def forward(self, x):
        # 提取多尺度特征
        feat1 = self.conv1(x)  # 小尺度特征
        feat2 = self.conv2(x)  # 中尺度特征
        feat3 = self.conv3(x)  # 大尺度特征

        # 特征融合
        fused_feat = torch.cat((feat1, feat2, feat3), dim=1)  # 连接特征
        fused_feat = F.relu(fused_feat)

        # 自注意力计算
        attention_weights = self._self_attention(fused_feat)

        # 动态调整
        # output = self.fc(attention_weights.view(fused_feat.size(0), fused_feat.size(1), -1).permute(0, 2, 1)).permute(0, 2, 1)  # 调整因子
        # output = output.reshape(feat1.size(0), feat1.size(1), feat1.size(2), -1)
        output = self.conv4(attention_weights)

        return output

    def _self_attention(self, x):
        # 计算自注意力权重
        b, c, h, w = x.size()
        query = x.view(b, c, -1)  # (batch_size, channels, height * width)
        key = x.view(b, c, -1)
        value = x.view(b, c, -1)

        attention_scores = torch.bmm(query.transpose(1, 2), key)  # (batch_size, height * width, height * width)
        attention_weights = F.softmax(attention_scores / (c ** 0.5), dim=-1)

        # 计算加权特征
        attention_output = torch.bmm(attention_weights, value.transpose(1, 2)).transpose(1, 2)
        attention_output = attention_output.view(b, c, h, w)  # (batch_size, channels, height, width)

        return attention_output


class DeformFSAS(nn.Module):
    def __init__(self, dim, bais):
        super(DeformFSAS, self).__init__()
        self.to_hidden = DeformConv2d(dim, dim * 6, kernel_size=3, padding=1)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=6*dim, bias=bais)
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bais)
        self.norm = nn.LayerNorm(dim * 2)
        self.patch_size = 4

    def forward(self, x):
        hidden = self.to_hidden(x)
        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)
        q_fft = torch.fft.rfft2(q.float())
        k_fft = torch.fft.rfft2(k.float())

        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = self.norm(out)

        output = v * out
        output = self.project_out(output)

        return output