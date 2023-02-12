import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np

##################################################################
# We borrow the linear transformer defined in LoFTR(CVPR-2021) for capturing global dependencies
def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1

class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()

##################################################################
class DyReLU_base(nn.Module):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLU_base, self).__init__()
        self.channels = channels
        self.k = k
        self.conv_type = conv_type
        assert self.conv_type in ['1d', '2d']

        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 2*k)
        self.sigmoid = nn.Sigmoid()

        self.register_buffer('lambdas', torch.Tensor([1.]*k + [0.5]*k).float())
        self.register_buffer('init_v', torch.Tensor([1.] + [0.]*(2*k - 1)).float())

    def get_relu_coefs(self, x):
        theta = torch.mean(x, axis=-1)
        if self.conv_type == '2d':
            theta = torch.mean(theta, axis=-1)
        theta = self.fc1(theta)
        theta = self.relu(theta)
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        return theta

    def forward(self, x):
        raise NotImplementedError

class DyReLU(DyReLU_base):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLU, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Linear(channels // reduction, 2*k*channels)

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)

        relu_coefs = theta.view(-1, self.channels, 2*self.k) * self.lambdas + self.init_v

        if self.conv_type == '1d':
            # BxCxL -> LxBxCx1
            x_perm = x.permute(2, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # LxBxCx2 -> BxCxL
            result = torch.max(output, dim=-1)[0].permute(1, 2, 0)

        elif self.conv_type == '2d':
            # BxCxHxW -> HxWxBxCx1
            x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # HxWxBxCx2 -> BxCxHxW
            result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)

        return result

##################################################################
class GroupEncoderBlock(nn.Module):
    def __init__(self, indim=128, outdim=128, bias=True):
        super(GroupEncoderBlock, self).__init__()
        self.conv_d1 = nn.Conv2d(indim, indim, kernel_size=3, stride=1, padding=1, groups=indim, bias=bias)
        self.conv_d2 = nn.Conv2d(indim, indim, kernel_size=5, stride=1, padding=2, groups=indim, bias=bias)
        self.conv_d3 = nn.Conv2d(indim, indim, kernel_size=7, stride=1, padding=3, groups=indim, bias=bias)
        self.mlp = nn.Sequential(
            nn.Conv2d(indim * 4, indim * 4, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(indim * 4, outdim//2, kernel_size=1, stride=1, padding=0, bias=bias)
        )
        self.semantic = nn.Sequential(
            LocalityChannelAttention(dim=indim, winsize=16),
            nn.Conv2d(indim, outdim//2, kernel_size=3, stride=1, padding=1, bias=bias)
        )

    def forward(self, x):
        x0 = self.conv_d1(x)
        x1 = self.conv_d2(x)
        x2 = self.conv_d3(x)
        x0 = torch.cat((x, x0, x1, x2), dim=1)
        del x1, x2
        x0 = self.mlp(x0)
        x = self.semantic(x)
        return torch.cat((x0, x), dim=1)

##################################################################
class FastSparseLinearSelfAttention(nn.Module):
    def __init__(self, dim=128, num_heads=8, kernel=4):
        super(FastSparseLinearSelfAttention, self).__init__()
        self.ndim = dim // num_heads
        self.nhead = num_heads
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.att = LinearAttention()
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=True)
        self.ks = kernel
        self.start = kernel//2

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.q(x)
        k, v = self.kv(x[:, :, self.start::self.ks, self.start::self.ks]).chunk(2, dim=1)
        q = rearrange(q, 'b (head c) h w -> b (h w) head c', head=self.nhead)
        k = rearrange(k, 'b (head c) h w -> b (h w) head c', head=self.nhead)
        v = rearrange(v, 'b (head c) h w -> b (h w) head c', head=self.nhead)
        q = self.att(q, k, v, q_mask=None, kv_mask=None)
        del k, v
        q = rearrange(q, 'b (h w) head c -> b (head c) h w', h=h, w=w)
        q = self.proj(q)
        return q

class FastTransformerHead(nn.Module):
    def __init__(self, dim=128, num_heads=8, kernel=4):
        super(FastTransformerHead, self).__init__()
        self.position_embedding = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.attn = FastSparseLinearSelfAttention(dim, num_heads, kernel)
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=2 * kernel - 1, padding=(2 * kernel - 2) // 2, groups=dim),
            nn.Conv2d(dim, dim, kernel_size=1)
        )
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1)
        )
        self.norm1 = nn.InstanceNorm2d(dim)
        self.norm2 = nn.InstanceNorm2d(dim)

    def forward(self, x):
        x = x + self.position_embedding(x)
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm) + self.local(x_norm)
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)
        return x

##################################################################
class HTNet(nn.Module):
    def __init__(self, dim=128):
        super(HTNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.PReLU(),
            nn.InstanceNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            DyReLU(channels=64),
            nn.InstanceNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            DyReLU(channels=128),
            nn.InstanceNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2),
            DyReLU(channels=128),
            nn.InstanceNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        )
        self.local1 = nn.Sequential(
            GroupEncoderBlock(indim=128, outdim=128, bias=True),
            nn.InstanceNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        )
        self.local2 = nn.Sequential(
            GroupEncoderBlock(indim=128, outdim=128, bias=True),
            nn.InstanceNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        )
        self.local3 = nn.Sequential(
            GroupEncoderBlock(indim=128, outdim=128, bias=True),
            nn.InstanceNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        )
        self.global1 = nn.Sequential(
            FastTransformerHead(dim=128, num_heads=8, kernel=4),
            FastTransformerHead(dim=128, num_heads=8, kernel=4)
        )
        self.global2 = nn.Sequential(
            FastTransformerHead(dim=128, num_heads=8, kernel=3),
            FastTransformerHead(dim=128, num_heads=8, kernel=3)
        )
        self.global3 = nn.Sequential(
            FastTransformerHead(dim=128, num_heads=8, kernel=2),
            FastTransformerHead(dim=128, num_heads=8, kernel=2)
        )

        self.head = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            DyReLU(channels=512),
            nn.InstanceNorm2d(512),
            nn.Conv2d(512, dim, kernel_size=1, stride=1, padding=0)
        )


    def forward(self, x):
        x = self.conv(x) # 0.25x 128
        x1 = self.global1(x)
        x = self.local1(x)
        x2 = self.global2(x)
        x = self.local2(x)
        x3 = self.global3(x)
        x = self.local3(x)
        x = self.head(torch.cat((x, x1, x2, x3), dim=1))
        return x

class LocalityChannelAttention(nn.Module):
    def __init__(self, dim=64, winsize=8):
        super(LocalityChannelAttention, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        )
        self.win_pool = nn.AvgPool2d(kernel_size=winsize, stride=winsize//2)
        #self.gate = nn.Sigmoid()

    def forward(self, x):
        h, w = x.shape[-2:]
        y = self.win_pool(x)
        y = self.mlp(y)
        # hard-sigmoid
        y = F.relu6(y + 3., inplace=True) / 6.
        y = F.interpolate(y, size=(h, w), mode='nearest')
        return x * y

##################################################################
class Descriptor(nn.Module):
    def __init__(self, dim=128):
        super(Descriptor, self).__init__()
        self.backbone = HTNet(dim=dim)

    def forward(self, x):
        x = self.backbone(x)
        return x

##################################################################
class HoGpredictor(nn.Module):
    def __init__(self, dim=128):
        super(HoGpredictor, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(dim // 2),
            nn.Conv2d(dim // 2, dim // 2, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(dim // 2),

        )
        self.body1 = nn.Sequential(
            nn.Conv2d(dim // 2, 36, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(36),
            nn.Conv2d(36, 36, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.body(x)
        x = F.interpolate(x, size=[14, 19], mode='bilinear', align_corners=True)
        x = self.body1(x)
        x = rearrange(x, 'b c h w -> b (w h c)') #14*19*36 #pytorch先叠最后一维
        return x

##########################################################################
## DeSTR
class DeSTR(nn.Module):
    def __init__(self, input_chan=1, d_model=128, nhead=8):
        super(DeSTR, self).__init__()

        self.descriptor = Descriptor(dim=128)
        self.hog = HoGpredictor(dim=128)

    def _forw_impl(self, x):
        return self.descriptor(x)

    def forward(self, x):
        return F.normalize(self._forw_impl(x), dim=1)

