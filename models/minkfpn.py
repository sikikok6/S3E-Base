# Author: Jacek Komorowski
# Warsaw University of Technology

import torch
import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock
import torchvision.models as models
from models.resnet import ResNetBase

# Modified by Defu LIN, on 20230430
from models.netvlad import NetVLADLoupe, NetVLAD

# import models
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, pack, unpack


class MinkFPN(ResNetBase):
    # Feature Pyramid Network (FPN) architecture implementation using Minkowski ResNet building blocks
    def __init__(self, in_channels, out_channels, num_top_down=1, conv0_kernel_size=5, block=BasicBlock,
                 layers=(1, 1, 1), planes=(32, 64, 64)):
        assert len(layers) == len(planes)
        assert 1 <= len(layers)
        assert 0 <= num_top_down <= len(layers)
        self.num_bottom_up = len(layers)
        self.num_top_down = num_top_down
        self.conv0_kernel_size = conv0_kernel_size
        self.block = block
        self.layers = layers
        self.planes = planes
        self.lateral_dim = out_channels
        self.init_dim = planes[0]
        ResNetBase.__init__(self, in_channels, out_channels, D=3)

    def network_initialization(self, in_channels, out_channels, D):
        assert len(self.layers) == len(self.planes)
        assert len(self.planes) == self.num_bottom_up

        self.convs = nn.ModuleList()    # Bottom-up convolutional blocks with stride=2
        self.bn = nn.ModuleList()       # Bottom-up BatchNorms
        self.blocks = nn.ModuleList()   # Bottom-up blocks
        self.tconvs = nn.ModuleList()   # Top-down tranposed convolutions
        self.conv1x1 = nn.ModuleList()  # 1x1 convolutions in lateral connections

        # The first convolution is special case, with kernel size = 5
        self.inplanes = self.planes[0]
        self.conv0 = ME.MinkowskiConvolution(in_channels, self.inplanes, kernel_size=self.conv0_kernel_size,
                                             dimension=D)
        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        for plane, layer in zip(self.planes, self.layers):
            self.convs.append(ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D))
            self.bn.append(ME.MinkowskiBatchNorm(self.inplanes))
            self.blocks.append(self._make_layer(self.block, plane, layer))

        # Lateral connections
        for i in range(self.num_top_down):
            self.conv1x1.append(ME.MinkowskiConvolution(self.planes[-1 - i], self.lateral_dim, kernel_size=1,
                                                        stride=1, dimension=D))
            self.tconvs.append(ME.MinkowskiConvolutionTranspose(self.lateral_dim, self.lateral_dim, kernel_size=2,
                                                                stride=2, dimension=D))
        # There's one more lateral connection than top-down TConv blocks
        if self.num_top_down < self.num_bottom_up:
            # Lateral connection from Conv block 1 or above
            self.conv1x1.append(ME.MinkowskiConvolution(self.planes[-1 - self.num_top_down], self.lateral_dim, kernel_size=1,
                                                        stride=1, dimension=D))
        else:
            # Lateral connection from Con0 block
            self.conv1x1.append(ME.MinkowskiConvolution(self.planes[0], self.lateral_dim, kernel_size=1,
                                                        stride=1, dimension=D))

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        # *** BOTTOM-UP PASS ***
        # First bottom-up convolution is special (with bigger stride)
        feature_maps = []
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)
        if self.num_top_down == self.num_bottom_up:
            feature_maps.append(x)

        # BOTTOM-UP PASS
        for ndx, (conv, bn, block) in enumerate(zip(self.convs, self.bn, self.blocks)):
            x = conv(x)     # Decreases spatial resolution (conv stride=2)
            x = bn(x)
            x = self.relu(x)
            x = block(x)
            if self.num_bottom_up - 1 - self.num_top_down <= ndx < len(self.convs) - 1:
                feature_maps.append(x)

        assert len(feature_maps) == self.num_top_down

        x = self.conv1x1[0](x)

        # TOP-DOWN PASS
        for ndx, tconv in enumerate(self.tconvs):
            x = tconv(x)        # Upsample using transposed convolution
            x = x + self.conv1x1[ndx+1](feature_maps[-ndx - 1])

        return x



class ResnetFPN(torch.nn.Module):
    def __init__(self, out_channels: int, lateral_dim: int, layers=[64, 64, 128, 256, 512], fh_num_bottom_up: int = 5,
                 fh_num_top_down: int = 2, add_fc_block: bool = False, pool_method='spoc'):
        # Pooling types:  GeM, sum-pooled convolution (SPoC), maximum activations of convolutions (MAC)
        super().__init__()
        assert 0 < fh_num_bottom_up <= 5
        assert 0 <= fh_num_top_down < fh_num_bottom_up

        self.out_channels = out_channels
        self.lateral_dim = lateral_dim
        self.fh_num_bottom_up = fh_num_bottom_up
        self.fh_num_top_down = fh_num_top_down
        self.add_fc_block = add_fc_block
        self.layers = layers    # Number of channels in output from each ResNet block
        self.pool_method = pool_method.lower()
        model = models.resnet18(pretrained=True)
        # Last 2 blocks are AdaptiveAvgPool2d and Linear (get rid of them)
        self.resnet_fe = nn.ModuleList(
            list(model.children())[:3+self.fh_num_bottom_up])

        # Lateral connections and top-down pass for the feature extraction head
        # Top-down transposed convolutions in feature head
        self.fh_tconvs = nn.ModuleDict()
        # 1x1 convolutions in lateral connections to the feature head
        self.fh_conv1x1 = nn.ModuleDict()
        for i in range(self.fh_num_bottom_up - self.fh_num_top_down, self.fh_num_bottom_up):
            self.fh_conv1x1[str(i + 1)] = nn.Conv2d(in_channels=layers[i],
                                                    out_channels=self.lateral_dim, kernel_size=1)
            self.fh_tconvs[str(i + 1)] = torch.nn.ConvTranspose2d(in_channels=self.lateral_dim,
                                                                  out_channels=self.lateral_dim,
                                                                  kernel_size=2, stride=2)

        # One more lateral connection
        temp = self.fh_num_bottom_up - self.fh_num_top_down
        self.fh_conv1x1[str(temp)] = nn.Conv2d(
            in_channels=layers[temp-1], out_channels=self.lateral_dim, kernel_size=1)

        # Pooling types:  GeM, sum-pooled convolution (SPoC), maximum activations of convolutions (MAC)
        if self.pool_method == 'gem':
            self.pool = GeM()
        elif self.pool_method == 'spoc':
            self.pool = nn.AdaptiveAvgPool2d((32, 32))
        elif self.pool_method == 'max':
            self.pool = nn.AdaptiveMaxPool2d((32, 32))
        else:
            raise NotImplementedError(
                "Unknown pooling method: {}".format(self.pool_method))

        if self.add_fc_block:
            self.fc = torch.nn.Linear(
                in_features=self.lateral_dim, out_features=self.out_channels)

    def forward(self, batch):
        x = batch['images']
        feature_maps = {}

        # 0, 1, 2, 3 = first layers: Conv2d, BatchNorm, ReLu, MaxPool2d
        x = self.resnet_fe[0](x)
        x = self.resnet_fe[1](x)
        x = self.resnet_fe[2](x)
        x = self.resnet_fe[3](x)
        feature_maps["1"] = x

        # sequential blocks, build from BasicBlock or Bottleneck blocks
        for i in range(4, self.fh_num_bottom_up+3):
            x = self.resnet_fe[i](x)
            feature_maps[str(i-2)] = x

        assert len(feature_maps) == self.fh_num_bottom_up
        # x is (batch_size, 512, H=20, W=15) for 640x480 input image

        # FEATURE HEAD TOP-DOWN PASS
        xf = self.fh_conv1x1[str(self.fh_num_bottom_up)](
            feature_maps[str(self.fh_num_bottom_up)])
        for i in range(self.fh_num_bottom_up, self.fh_num_bottom_up - self.fh_num_top_down, -1):
            # Upsample using transposed convolution
            xf = self.fh_tconvs[str(i)](xf)
            xf = xf + self.fh_conv1x1[str(i-1)](feature_maps[str(i - 1)])

        # x = self.pool(xf)
        # x is (batch_size, 512, 1, 1) tensor

        # x = torch.flatten(x, 1)
        # x is (batch_size, 512) tensor

        # if self.add_fc_block:
        #     x = self.fc(x)

        # (batch_size, feature_size)
        # assert x.shape[1] == self.out_channels
        return x


# GeM code adapted from: https://github.com/filipradenovic/cnnimageretrieval-pytorch

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return nn.functional.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

'''

class MinkFPN(ResNetBase):
    # Feature Pyramid Network (FPN) architecture implementation using Minkowski ResNet building blocks
    def __init__(self, in_channels, out_channels, num_top_down=1, conv0_kernel_size=5, block=BasicBlock,
                 layers=(1, 1, 1), planes=(32, 64, 64)):
        assert len(layers) == len(planes)
        assert 1 <= len(layers)
        assert 0 <= num_top_down <= len(layers)
        self.num_bottom_up = len(layers)
        self.num_top_down = num_top_down
        self.conv0_kernel_size = conv0_kernel_size
        self.block = block
        self.layers = layers
        self.planes = planes
        self.lateral_dim = out_channels
        self.init_dim = planes[0]
        ResNetBase.__init__(self, in_channels, out_channels, D=3)

    def network_initialization(self, in_channels, out_channels, D):
        assert len(self.layers) == len(self.planes)
        assert len(self.planes) == self.num_bottom_up

        self.convs = nn.ModuleList()    # Bottom-up convolutional blocks with stride=2
        self.bn = nn.ModuleList()       # Bottom-up BatchNorms
        self.blocks = nn.ModuleList()   # Bottom-up blocks
        self.tconvs = nn.ModuleList()   # Top-down tranposed convolutions
        self.conv1x1 = nn.ModuleList()  # 1x1 convolutions in lateral connections

        # The first convolution is special case, with kernel size = 5
        self.inplanes = self.planes[0]
        self.conv0 = ME.MinkowskiConvolution(in_channels, self.inplanes, kernel_size=self.conv0_kernel_size,
                                             dimension=D)
        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        for plane, layer in zip(self.planes, self.layers):
            self.convs.append(ME.MinkowskiConvolution(
                self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D))
            self.bn.append(ME.MinkowskiBatchNorm(self.inplanes))
            self.blocks.append(self._make_layer(self.block, plane, layer))

        # Lateral connections
        for i in range(self.num_top_down):
            self.conv1x1.append(ME.MinkowskiConvolution(self.planes[-1 - i], self.lateral_dim, kernel_size=1,
                                                        stride=1, dimension=D))
            self.tconvs.append(ME.MinkowskiConvolutionTranspose(self.lateral_dim, self.lateral_dim, kernel_size=2,
                                                                stride=2, dimension=D))
        # There's one more lateral connection than top-down TConv blocks
        if self.num_top_down < self.num_bottom_up:
            # Lateral connection from Conv block 1 or above
            self.conv1x1.append(ME.MinkowskiConvolution(self.planes[-1 - self.num_top_down], self.lateral_dim, kernel_size=1,
                                                        stride=1, dimension=D))
        else:
            # Lateral connection from Con0 block
            self.conv1x1.append(ME.MinkowskiConvolution(self.planes[0], self.lateral_dim, kernel_size=1,
                                                        stride=1, dimension=D))

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        # *** BOTTOM-UP PASS ***
        # First bottom-up convolution is special (with bigger stride)
        feature_maps = []
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)
        if self.num_top_down == self.num_bottom_up:
            feature_maps.append(x)

        # BOTTOM-UP PASS
        for ndx, (conv, bn, block) in enumerate(zip(self.convs, self.bn, self.blocks)):
            x = conv(x)     # Decreases spatial resolution (conv stride=2)
            x = bn(x)
            x = self.relu(x)
            x = block(x)
            if self.num_bottom_up - 1 - self.num_top_down <= ndx < len(self.convs) - 1:
                feature_maps.append(x)

        assert len(feature_maps) == self.num_top_down

        x = self.conv1x1[0](x)

        # TOP-DOWN PASS
        for ndx, tconv in enumerate(self.tconvs):
            x = tconv(x)        # Upsample using transposed convolution
            x = x + self.conv1x1[ndx+1](feature_maps[-ndx - 1])

        return x

'''
###########################################################################
# Net
###########################################################################


class ResnetFPN_M(torch.nn.Module):
    def __init__(self, out_channels: int, lateral_dim: int, layers=[64, 64, 128, 256, 512], fh_num_bottom_up: int = 5,
                 fh_num_top_down: int = 2, add_fc_block: bool = False, pool_method='spoc'):
        # Pooling types:  GeM, sum-pooled convolution (SPoC), maximum activations of convolutions (MAC)
        super().__init__()
        assert 0 < fh_num_bottom_up <= 5
        assert 0 <= fh_num_top_down < fh_num_bottom_up

        self.out_channels = out_channels
        self.lateral_dim = lateral_dim
        # 自下往上（特征提取），resnet层数，一层为一个basicblock（四个卷积，一种颜色）
        self.fh_num_bottom_up = fh_num_bottom_up
        self.fh_num_top_down = fh_num_top_down
        self.add_fc_block = add_fc_block
        self.layers = layers  # Number of channels in output from each ResNet block
        self.pool_method = pool_method.lower()
        model = models.resnet18(pretrained=True)
        # Last 2 blocks are AdaptiveAvgPool2d and Linear (get rid of them)
        self.resnet_fe = nn.ModuleList(
            list(model.children())[:3 + self.fh_num_bottom_up])

        # Lateral connections and top-down pass for the feature extraction head
        # Top-down transposed convolutions in feature head
        self.fh_tconvs = nn.ModuleDict()
        # 1x1 convolutions in lateral connections to the feature head
        self.fh_conv1x1 = nn.ModuleDict()
        self.last_conv = nn.Conv2d(in_channels=self.lateral_dim, out_channels=self.lateral_dim,
                                   kernel_size=3, stride=2, padding=1)
        for i in range(self.fh_num_bottom_up - self.fh_num_top_down, self.fh_num_bottom_up):  # 3,4
            # 1x1卷积:通道全部变为128个通道
            self.fh_conv1x1[str(i + 1)] = nn.Conv2d(in_channels=layers[i], out_channels=self.lateral_dim,
                                                    kernel_size=1)  # 3,4
            # 转置卷积：上采样：k x k -> 2k x 2k， 通道为128不变
            if i + 1 in [4, 3]:
                self.fh_tconvs[str(i + 1)] = torch.nn.ConvTranspose2d(in_channels=self.lateral_dim,  # 4,3
                                                                      out_channels=self.lateral_dim,
                                                                      kernel_size=2, stride=2)
            elif i + 1 <= 2:
                print('please write fh_tconvs')
            else:  # 5
                self.fh_tconvs[str(i + 1)] = torch.nn.ConvTranspose2d(in_channels=self.lateral_dim,
                                                                      out_channels=self.lateral_dim,
                                                                      kernel_size=2, stride=2)

        # One more lateral connection
        temp = self.fh_num_bottom_up - self.fh_num_top_down
        self.fh_conv1x1[str(temp)] = nn.Conv2d(in_channels=layers[temp - 1], out_channels=self.lateral_dim,
                                               kernel_size=1)
        # 5- 2的输入:
        # [[3  128]
        # [4  256]
        # [5  512]]

        # Pooling types:  GeM, sum-pooled convolution (SPoC), maximum activations of convolutions (MAC)
        if self.pool_method == 'gem':
            self.pool = GeM()
        elif self.pool_method == 'spoc':
            self.pool = nn.AdaptiveAvgPool2d((32, 32))
        elif self.pool_method == 'max':
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise NotImplementedError(
                "Unknown pooling method: {}".format(self.pool_method))

        if self.add_fc_block:
            self.fc = torch.nn.Linear(
                in_features=self.lateral_dim, out_features=self.out_channels)

    def forward(self, batch):
        x = batch['images']  # ['images']
        feature_maps = {}

        # 0, 1, 2, 3 = first layers: Conv2d, BatchNorm, ReLu, MaxPool2d
        x = self.resnet_fe[0](x)
        x = self.resnet_fe[1](x)
        x = self.resnet_fe[2](x)
        x = self.resnet_fe[3](x)
        feature_maps["1"] = x

        # sequential blocks, build from BasicBlock or Bottleneck blocks # 自底向上，每层是一个basicblock
        for i in range(4, self.fh_num_bottom_up + 3):
            x = self.resnet_fe[i](x)
            feature_maps[str(i - 2)] = x
        # 5-2输入：
        # resnet_fe：0~4~7， feature_map：1~5
        # 4输入：
        # resnet_fe：0~4~6，feature_map：1~4

        assert len(feature_maps) == self.fh_num_bottom_up
        # x is (batch_size, 512, H=20, W=15) for 640x480 input image
        # 没有0层[[64,125],[64,125],[128，63]，[256，32]，[512，116]](5层resnet18)
        # x is (batch_size, 512, H=16, W=16) for 500x500 input image

        # FEATURE HEAD TOP-DOWN PASS # 上采样并和上面的feature_map相加，2次自顶向下
        xf = self.fh_conv1x1[str(self.fh_num_bottom_up)](
            feature_maps[str(self.fh_num_bottom_up)])
        for i in range(self.fh_num_bottom_up, self.fh_num_bottom_up - self.fh_num_top_down,
                       -1):  # tconv:5, 4, 1x1conv:4.3+
            # Upsample using transposed convolution
            xf = self.fh_tconvs[str(i)](xf)
            # 出错，63，64
            xf = xf + self.fh_conv1x1[str(i - 1)](feature_maps[str(i - 1)])

        # xf is (batch_size, 128, 63, 63) tensor -> (batch_size, 128, 32, 32)
        x = self.last_conv(xf)
        # x is (batch_size, 512, 1, 1) tensor
        x = self.pool(x)

        # x = torch.flatten(x, 1)
        '''x = torch.flatten(x, 1)
        # x is (batch_size, 512) tensor

        if self.add_fc_block:
            x = self.fc(x)

        # (batch_size, feature_size)
        assert x.shape[1] == self.out_channels'''
        return x


class GeM(nn.Module):  # GeM code adapted from: https://github.com/filipradenovic/cnnimageretrieval-pytorch
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return nn.functional.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1. / self.p)

# Vit-1-d


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads,
                        dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * \
            (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class ResnetFPN_T(nn.Module):
    def __init__(self, out_channels: int, lateral_dim: int):
        super().__init__()
        self.out_channels = out_channels
        self.lateral_dim = lateral_dim
        self.resnetfpn = ResnetFPN(
            out_channels, out_channels, fh_num_bottom_up=4, fh_num_top_down=0)
        # image_fe = ResnetFPN(out_channels=image_fe_size, lateral_dim=image_fe_size,
        #                      fh_num_bottom_up=4, fh_num_top_down=0)
        # self.resnetfpn = ResnetFPN_M(self.out_channels, self.lateral_dim)
        self.vit = ViT(
            image_size=32,
            patch_size=32,
            num_classes=128,
            dim=1024,  # 1024->256
            depth=6,
            heads=12,
            mlp_dim=2048,  # 2048->256
            channels=128,
            dropout=0.1,
            emb_dropout=0.1
        )

    def forward(self, batch):
        x = self.resnetfpn(batch)  # x is (batch_size, 128, 32, 32)
        # x = torch.reshape(x, (x.shape[0], 1, x.shape[1]))
        x = self.vit(x)
        # assert x.shape[1] == self.out_channels
        return x


# channels can be 15
class ResnetFPN_vlad(nn.Module):
    def __init__(self, out_channels: int = 128, lateral_dim: int = 128, channels=32, norm_layer=None, use_transformer=True,
                 layers=[64, 64, 128, 256, 512], fh_num_bottom_up: int = 5, fh_num_top_down: int = 2, add_fc_block: bool = False, pool_method='spoc'):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d   # number of channels

        self.out_channels = out_channels
        self.laterral_dim = lateral_dim
        self.fh_num_bottom_up = fh_num_bottom_up
        self.fh_num_top_down = fh_num_top_down
        self.use_transformer = use_transformer
        # self.resnetfpn = ResnetFPN(out_channels, out_channels, fh_num_bottom_up=4, fh_num_top_down=0)
        # [16, 256, 30, 40]
        # NeVLADLoupe will take a shape = [a, b, c, d] four dimensional input
        model = models.resnet18(pretrained=True)
        self.resnet_fe = nn.ModuleList(
            list(model.children())[:3+self.fh_num_bottom_up])
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=(
            2, 1), stride=(2, 1), bias=False)
        self.bn1 = norm_layer(16)
        self.conv1_add = nn.Conv2d(
            16, 16, kernel_size=(5, 1), stride=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(
            3, 1), stride=(1, 1), bias=False)
        self.bn2 = norm_layer(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(
            3, 1), stride=(1, 1), bias=False)
        self.bn3 = norm_layer(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(
            3, 1), stride=(1, 1), bias=False)
        self.bn4 = norm_layer(64)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(
            3, 1), stride=(1, 1), bias=False)
        self.bn5 = norm_layer(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=(
            3, 1), stride=(1, 1), bias=False)
        self.bn6 = norm_layer(128)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=(
            1, 1), stride=(2, 1), bias=False)
        self.bn7 = norm_layer(128)  # now the output will with 128 dimension

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=4, dim_feedforward=1024, activation='relu', batch_first=False, dropout=0.)
        # # This is a transformerencoder API provided by torch
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=1)  # 3 6

        self.relu = nn.ReLU(inplace=True)

        self.convLast1 = nn.Conv2d(
            128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        # now the output will with 256 dimension
        self.bnLast1 = norm_layer(256)
        self.convLast2 = nn.Conv2d(
            512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        # now the output will with 1024 dimension
        self.bnLast2 = norm_layer(1024)

        # in_features = 128 * width, out_features = 256
        self.linear = nn.Linear(128*900, 256)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        # max_sample can be 2420
        self.net_vlad = NetVLADLoupe(feature_size=1024, max_samples=800, cluster_size=64,  # before 11.12 --- 64
                                     output_dim=self.out_channels, gating=True, add_batch_norm=False,   # output_dim=512
                                     is_training=True)

        self.linear1 = nn.Linear(1 * 256, 256)
        self.bnl1 = norm_layer(256)
        self.linear2 = nn.Linear(1 * 256, 256)
        self.bnl2 = norm_layer(256)
        self.linear3 = nn.Linear(1 * 256, 256)
        self.bnl3 = norm_layer(256)

    def forward(self, batch):
        # x = batch['images']
        # feature_maps = {}

        # 0, 1, 2, 3 = first layers: Conv2d, BatchNorm, ReLu, MaxPool2d
        # x = self.resnet_fe[0](x)
        # x = self.resnet_fe[1](x)
        # x = self.resnet_fe[2](x)
        # x = self.resnet_fe[3](x)
        # feature_maps["1"] = x

        # sequential blocks, build from BasicBlock or Bottleneck blocks
        # for i in range(4, self.fh_num_bottom_up+3):
        #   x = self.resnet_fe[i](x)
        #   feature_maps[str(i-2)] = x
        x = self.resnetfpn(batch)  # x is (batch_size, 128, 32, 32)
        # print("01x.shape: {}".format(x.shape))
        x = x.permute(0, 2, 1, 3)
        # print("02x.shape: {}".format(x.shape))
        out_l = self.relu(self.conv1(x))
        # print("01out_l.shape: {}".format(out_l.shape))
        out_l = self.relu(self.conv1_add(out_l))
        # print("02out_l.shape: {}".format(out_l.shape))
        out_l = self.relu(self.conv2(out_l))
        # print("03out_l.shape: {}".format(out_l.shape))
        out_l = self.relu(self.conv3(out_l))
        # print("04out_l.shape: {}".format(out_l.shape))
        out_l = self.relu(self.conv4(out_l))
        # print("05out_l.shape: {}".format(out_l.shape))
        out_l = self.relu(self.conv5(out_l))
        # print("06out_l.shape: {}".format(out_l.shape))
        out_l = self.relu(self.conv6(out_l))
        # print("07out_l.shape: {}".format(out_l.shape))
        out_l = self.relu(self.conv7(out_l))  # the out_channel will be 128
        # print("08out_l.shape: {}".format(out_l.shape))

        # out_r (bs, 128,360, 1) ## rearrange the dimension with given order
        out_l_1 = out_l.permute(0, 1, 3, 2)
        out_l_1 = self.relu(self.convLast1(out_l_1))  # out_channel = 256

        if self.use_transformer:
            # squeeze() function can delete the 1-dim like squeeze(3): [bs, 256, 360, 1] -> [bs, 256, 360]
            _0, _1, _2, _3 = out_l_1.shape
            out_l = out_l_1.reshape(_0, _1, _2*_3)
            # print("09out_l.shape: {}".format(out_l.shape))
            # [bs, 128, 360] -> [360, bs, 256]
            out_l = out_l.permute(2, 0, 1)
            out_l = self.transformer_encoder(out_l)

            out_l = out_l.permute(1, 2, 0)  # [360, bs, 256] -> [bs, 256, 360]
            # print("10out_l.shape: {}".format(out_l.shape))
            out_l = out_l.unsqueeze(3)  # [bs, 256, 360] -> [bs, 256, 360, 1]
            # print("11out_l.shape: {}".format(out_l.shape))
            out_l_1 = (out_l_1.reshape(_0, _1, _2*_3)).unsqueeze(3)
            print("11out_l_1.shape: {}".format(out_l_1.shape))
            out_l = torch.cat((out_l_1, out_l), dim=1)  # [bs, 512, 360, 1]
            out_l = self.relu(self.convLast2(out_l))  #
            # torch.nn.functional.normalize(input, p=2, dim=1, eps=1e-12, out=None) ## # return the L-p norm of given dimention
            # shape = [bs, 256(normalized), 360, 1]
            out_l = nn.functional.normalize(out_l, dim=1)

            # This is the part we want
            print("12out_l.shape: {}".format(out_l.shape))
            out_l = self.net_vlad(out_l)  # out_dim = 256

            out_l = nn.functional.normalize(out_l, dim=1)

        return out_l

class Resnet3DFPN_df(torch.nn.Module):
    def __init__(self, out_channels: int, lateral_dim: int, layers=[64, 64, 128, 256, 512], fh_num_bottom_up: int = 5,
                 fh_num_top_down: int = 2, add_fc_block: bool = False, pool_method='gem'):
        # Pooling types:  GeM, sum-pooled convolution (SPoC), maximum activations of convolutions (MAC)
        super().__init__()
        assert 0 < fh_num_bottom_up <= 5
        assert 0 <= fh_num_top_down < fh_num_bottom_up

        self.out_channels = out_channels
        self.lateral_dim = lateral_dim
        self.fh_num_bottom_up = fh_num_bottom_up
        self.fh_num_top_down = fh_num_top_down
        self.add_fc_block = add_fc_block
        self.layers = layers    # Number of channels in output from each ResNet block
        self.pool_method = pool_method.lower()

        """This part need to be change"""
        model = models.video.r3d_18(weights='DEFAULT')
        # Last 2 blocks are AdaptiveAvgPool2d and Linear (get rid of them)
        self.resnet_fe = nn.ModuleList(
            list(model.children())[:self.fh_num_bottom_up+1])

        # Lateral connections and top-down pass for the feature extraction head
        # Top-down transposed convolutions in feature head
        self.fh_tconvs = nn.ModuleDict()
        # 1x1 convolutions in lateral connections to the feature head
        self.fh_conv1x1 = nn.ModuleDict()
        for i in range(self.fh_num_bottom_up - self.fh_num_top_down, self.fh_num_bottom_up):
            self.fh_conv1x1[str(i + 1)] = nn.Conv3d(in_channels=layers[i],
                                                    out_channels=self.lateral_dim, kernel_size=1)
            self.fh_tconvs[str(i + 1)] = torch.nn.ConvTranspose3d(in_channels=self.lateral_dim,
                                                                  out_channels=self.lateral_dim,
                                                                  kernel_size=2, stride=2)

        # One more lateral connection
        temp = self.fh_num_bottom_up - self.fh_num_top_down
        self.fh_conv1x1[str(temp)] = nn.Conv3d(
            in_channels=layers[temp-1], out_channels=self.lateral_dim, kernel_size=1)
        """The change part end"""

        # Pooling types:  GeM, sum-pooled convolution (SPoC), maximum activations of convolutions (MAC)
        if self.pool_method == 'gem':
            self.pool = GeM()
        elif self.pool_method == 'spoc':
            self.pool = nn.AdaptiveAvgPool2d((32, 32))
        elif self.pool_method == 'max':
            self.pool = nn.AdaptiveMaxPool2d((32, 32))
        else:
            raise NotImplementedError(
                "Unknown pooling method: {}".format(self.pool_method))

        if self.add_fc_block:
            self.fc = torch.nn.Linear(
                in_features=self.lateral_dim, out_features=self.out_channels)

    def forward(self, batch):
        x = batch['images']
        feature_maps = {}
        _bz = x.shape[0]
        x = x.unsqueeze(4)
        x = x.permute(0, 1, 4, 2, 3)
        # print(f"x01.shape: {x.shape}, bz: {_bz}")
        # 0, 1, 2, 3 = first layers: Conv2d, BatchNorm, ReLu, MaxPool2d
        x = self.resnet_fe[0](x)
        x = self.resnet_fe[1](x)
        feature_maps["1"] = x
        x = self.resnet_fe[2](x)
        feature_maps["2"] = x
        x = self.resnet_fe[3](x)
        feature_maps["3"] = x

        # sequential blocks, build from BasicBlock or Bottleneck blocks
        for i in range(4, self.fh_num_bottom_up+1):
            x = self.resnet_fe[i](x)
            _03 = x.shape[3] * 2
            x = x.reshape(_bz, 256, 1, int(_03), 40)
            # print(f"x.shape: {x.shape}")
            feature_maps[str(i)] = x
            # print(f"feature_map[4]: {x.shape}")

        assert len(feature_maps) == self.fh_num_bottom_up
        # x is (batch_size, 512, H=20, W=15) for 640x480 input image

        # FEATURE HEAD TOP-DOWN PASS
        # print(feature_maps[str(self.fh_num_bottom_up)].reshape(1, 256, 4, 30, 40).shape)
        xf = self.fh_conv1x1[str(self.fh_num_bottom_up)](
            feature_maps[str(self.fh_num_bottom_up)])
        for i in range(self.fh_num_bottom_up, self.fh_num_bottom_up - self.fh_num_top_down, -1):
            # Upsample using transposed convolution
            xf = self.fh_tconvs[str(i)](xf)
            # print(f"xf00{i}.shape: {xf.shape}")
            xf = xf + self.fh_conv1x1[str(i-1)](feature_maps[str(i - 1)])
            # print(f"xf01{i}.shape: {xf.shape}")
        # print(f"xf.shape: {xf.shape}")

        xf = xf.squeeze(2)

        # print(f"xf_squeeze: {xf.shape}")

        x = self.pool(xf)
        # print(f"x_pool: {x.shape}")
        # x is (batch_size, 512, 1, 1) tensor

        x = torch.flatten(x, 1)
        # print(f"x_: {x.shape}")

        if self.add_fc_block:
            x = self.fc(x)
            # print(f"x_fc: {x.shape}")

        # (batch_size, feature_size)
        assert x.shape[0] == _bz
        assert x.shape[1] == self.out_channels
        return x


class Resnet3DFPN(torch.nn.Module):
    def __init__(self, out_channels: int, lateral_dim: int, layers=[64, 64, 128, 256, 512], fh_num_bottom_up: int = 5,
                 fh_num_top_down: int = 2, add_fc_block: bool = True, pool_method='gem'):
        # Pooling types:  GeM, sum-pooled convolution (SPoC), maximum activations of convolutions (MAC)
        super().__init__()
        assert 0 < fh_num_bottom_up <= 5
        assert 0 <= fh_num_top_down < fh_num_bottom_up

        self.out_channels = out_channels
        self.lateral_dim = lateral_dim
        self.fh_num_bottom_up = fh_num_bottom_up
        self.fh_num_top_down = fh_num_top_down
        self.add_fc_block = add_fc_block
        self.layers = layers    # Number of channels in output from each ResNet block
        self.pool_method = pool_method.lower()

        """This part need to be change"""
        model = models.video.r3d_18(weights='DEFAULT')
        # Last 2 blocks are AdaptiveAvgPool2d and Linear (get rid of them)
        self.resnet_fe = nn.ModuleList(
            list(model.children())[:self.fh_num_bottom_up+1])

        # Lateral connections and top-down pass for the feature extraction head
        # Top-down transposed convolutions in feature head
        self.fh_tconvs = nn.ModuleDict()
        # 1x1 convolutions in lateral connections to the feature head
        self.fh_conv1x1 = nn.ModuleDict()
        ks = [1, 2, 1, 1, 3]
        for i in range(self.fh_num_bottom_up - self.fh_num_top_down, self.fh_num_bottom_up):
            self.fh_conv1x1[str(i + 1)] = nn.Conv3d(in_channels=layers[i+1],
                                                    out_channels=self.lateral_dim, kernel_size=[ks[i], 1, 1])
            self.fh_tconvs[str(i + 1)] = torch.nn.ConvTranspose3d(in_channels=self.lateral_dim,
                                                                  out_channels=self.lateral_dim,
                                                                  kernel_size=[1, 2, 2], stride=2)

        # One more lateral connection
        temp = self.fh_num_bottom_up - self.fh_num_top_down
        self.fh_conv1x1[str(temp)] = nn.Conv3d(
            in_channels=layers[temp], out_channels=self.lateral_dim, kernel_size=[3, 1, 1])
        """The change part end"""

        # Pooling types:  GeM, sum-pooled convolution (SPoC), maximum activations of convolutions (MAC)
        if self.pool_method == 'gem':
            self.pool = GeM()
        elif self.pool_method == 'spoc':
            self.pool = nn.AdaptiveAvgPool2d((32, 32))
        elif self.pool_method == 'max':
            self.pool = nn.AdaptiveMaxPool2d((32, 32))
        else:
            raise NotImplementedError(
                "Unknown pooling method: {}".format(self.pool_method))

        if self.add_fc_block:
            self.fc = torch.nn.Linear(
                in_features=self.lateral_dim, out_features=self.out_channels)

    def forward(self, batch):
        x = batch['images']
        x = rearrange(x, 'N D C H W -> N C D H W')
        # x = batch

        feature_maps = {}
        _bz = x.shape[0]
        # x = x.unsqueeze(2)

        # print(f"x01.shape: {x.shape}, bz: {_bz}")
        # 0, 1, 2, 3 = first layers: Conv2d, BatchNorm, ReLu, MaxPool2d
        x = self.resnet_fe[0](x)
        # print(f"x00.shape {x.shape}")
        x = self.resnet_fe[1](x)
        feature_maps["1"] = x
        # print(f"x01.shape {x.shape}")
        x = self.resnet_fe[2](x)
        feature_maps["2"] = x
        # print(f"x02.shape {x.shape}")
        x = self.resnet_fe[3](x)
        feature_maps["3"] = x
        # print(f"x03.shape {x.shape}")

        # sequential blocks, build from BasicBlock or Bottleneck blocks
        for i in range(4, self.fh_num_bottom_up+1):
            x = self.resnet_fe[i](x)
            # _03 = x.shape[3] * 2
            # x = x.reshape(_bz, 256, 1, int(_03), 40)
            # print(f"x.shape: {x.shape}")
            feature_maps[str(i)] = x
            # print(f"feature_map[4]: {x.shape}")

        assert len(feature_maps) == self.fh_num_bottom_up
        # x is (batch_size, 512, H=20, W=15) for 640x480 input image

        # FEATURE HEAD TOP-DOWN PASS
        # print(feature_maps[str(self.fh_num_bottom_up)].reshape(1, 256, 4, 30, 40).shape)
        xf = self.fh_conv1x1[str(self.fh_num_bottom_up)](
            feature_maps[str(self.fh_num_bottom_up)])
        for i in range(self.fh_num_bottom_up, self.fh_num_bottom_up - self.fh_num_top_down, -1):
            # Upsample using transposed convolution
            xf = self.fh_tconvs[str(i)](xf)
            xf = xf + self.fh_conv1x1[str(i-1)](feature_maps[str(i - 1)])

        xf = xf.squeeze(2)

        # print(f"xf_squeeze: {xf.shape}")

        x = self.pool(xf)
        # print(f"x_pool: {x.shape}")
        # x is (batch_size, 512, 1, 1) tensor

        x = torch.flatten(x, 1)
        # print(f"x_: {x.shape}")

        if self.add_fc_block:
            x = self.fc(x)
            # print(f"x_fc: {x.shape}")

        # (batch_size, feature_size)
        assert x.shape[0] == _bz
        assert x.shape[1] == self.out_channels
        return x
