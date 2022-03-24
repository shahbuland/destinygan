import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from constants import *
import util
from torch_utils.ops import upfirdn2d, conv2d_gradfix, conv2d_resample

# Gets FIR kernel/filter
def get_fir_k():
    return upfirdn2d.setup_filter([1, 3, 3, 1]).cuda()

# Simplified wrapper for upfirdn2d
# for pooling and upsampling
class upFirDn2D(nn.Module):
    # kernel size of associated convolution
    # and factor by which to upsample (down if < 1)
    # (should be power of 2)
    def __init__(self, conv_k, scale = 1):
        super().__init__()
        
        # FIR filter used in nvidia implementation
        self.f = torch.Tensor([1, 3, 3, 1]).cuda()
        self.f = util.norm_fir_k(self.f)
        # Get needed padding for desired scale
        if scale == 0.5: # Downsample
            p = 1 + conv_k
            padX = (p + 1) // 2
            padY = p // 2
        elif scale == 1: # Same
            p = 3 - conv_k
            padX = (p + 1) // 2
            padY = p // 2
        elif scale > 1: # upsample
            p = 3 - conv_k
            padX = (p + 1) // 2 + 1
            padY = p // 2 + 1
            self.f *= (scale ** 2)
        
        self.p = (padX, padY)

    def forward(self, x):
        return upfirdn2d.upfirdn2d(x, self.f, padding = self.p)

# Bilinear downsampling using F.interpolate
class DownsamplingBilinear2d(nn.Module):
    def __init__(self, down_factor = 2):
        super().__init__()
        self.down_factor = down_factor

    def forward(self, x):
        _, _, h, w = x.shape
        new_h = h // self.down_factor
        new_w = w // self.down_factor
        return F.interpolate(x, size = (new_h, new_w), mode = 'bilinear',
                align_corners = False)
    
# Modified linear layer that scales weights
# Stylegan paper says they use 100x lower LR in mapping network, hence lr_mult
class modLinear(nn.Module):
    def __init__(self, dim_in, dim_out, lr_mult = 1, use_bias = True, use_act = False):
        super().__init__()

        self.lr_mult = lr_mult
        self.w = nn.Parameter(torch.randn(dim_out, dim_in).div_(lr_mult))
        self.bias = nn.Parameter(torch.zeros(dim_out)) if use_bias else None
        self.w_scale = lr_mult * dim_in**-.5
        self.act = nn.LeakyReLU(0.2) if use_act else None

    def forward(self, x):
        x = F.linear(x, self.w * self.w_scale, bias = self.bias * self.lr_mult)
        if self.act is not None: x = self.act(x)
        return x

# Conv layers for the discriminator
class discConv(nn.Module):
    # Mode can be "DOWN" or None
    def __init__(self, fi, fo, k, mode = None, use_bias = True, use_act = True):
        super().__init__()
        
        self.use_bias = use_bias
        self.use_act = use_act

        self.w = nn.Parameter(torch.randn(fo, fi, k, k))
        self.w_scale = (fi * k * k)**-.5
        self.b = nn.Parameter(torch.zeros(1, fo, 1, 1)) if use_bias else None
        self.act = nn.LeakyReLU(0.2) if use_act else None
        self.p = k // 2

        self.fir = get_fir_k()
        self.up = 2 if mode == "UP" else 1
        self.down = 2 if mode == "DOWN" else 1
    def forward(self, x):
        flip_weight = (self.up == 1)
        x = conv2d_resample.conv2d_resample(x = x, w = self.w * self.w_scale, f = self.fir,
                up = self.up, down = self.down, padding = self.p, flip_weight = flip_weight) 
        if self.use_bias: x = x + self.b
        if self.use_act: x = self.act(x) 
        return x

# Residual layer for discriminator
class DiscBlock(nn.Module):
    def __init__(self, fi, fo):
       super().__init__()

       self.conv1 = discConv(fi, fi, 3)
       self.conv2 = discConv(fi, fo, 3, mode = "DOWN")

       self.skip = discConv(fi, fo, 1, mode = "DOWN", use_bias = False, use_act = False)

    def forward(self, x):
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return (x + residual) / (2**.5)
