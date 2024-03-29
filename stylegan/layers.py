import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import einops as eo

from torch_utils.ops import upfirdn2d, conv2d_gradfix, conv2d_resample

from constants import *
import util

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

class UpsamplingBilinear2d(nn.Module):
    def __init__(self, up_factor):
        super().__init__()
        self.up_factor = up_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor = 2, mode = 'bilinear',
                align_corners = False)

# Normalization before mapping network
class mappingNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(Self, x):
        return x * torch.rsqrt(torch.mean(x.pow(2), dim = 1, keepdim = True) + EPSILON)

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

# For input to generator convolutional layers
class constantIn(nn.Module):
    def __init__(self, fi, k):
        super().__init__()

        self.inp_noise = nn.Parameter(torch.randn(1, fi, k, k))

    def forward(self, x):
        n = x.shape[0]
        if RANDOM_INP: y = torch.randn(n, 512, 4, 4, device = 'cuda')
        else: y = self.inp_noise.repeat(n, 1, 1, 1)
        
        return y

class simplifyIn(nn.Module):
    def __init__(self, fi, k):
        super().__init__()

        # need to map style vector (n, LATENT_DIM)
        # to n, 512, 4, 4
        self.map = nn.Linear(LATENT_DIM, fi * k * k)
        self.fi = fi
        self.k = k

    def forward(self, x):
        x = self.map(x)
        x = x.view(-1, self.fi, self.k, self.k)
        return x

# Modulated convolution
class modConv(nn.Module):
    # Mode can be "UP", "DOWN" or None
    def __init__(self, fi, fo, k, dim_style, mode = None, do_demod = True):
        super().__init__()

        self.fi = fi # Filters/channels in
        self.fo = fo # Filters/channels out
        self.k = k # Kernel size
        self.pad = k // 2
        self.mode = mode 
        self.do_demod = do_demod

        self.w = nn.Parameter(torch.randn(1, fo, fi, k, k))# Weight matrix
        self.w_scale = (fi * k * k)**-.5
        # Maps from style latent space to fi
        self.mod_fc = modLinear(dim_style, fi)
       
        self.fir = get_fir_k()
        self.up = 2 if mode == "UP" else 1
        self.down = 1
        #self.fir = upFirDn2D(k, scale = 2 if mode == "UP" else 1)
        #self.fir = UpsamplingBilinear2d(2) if mode == "UP" else DownsamplingBilinear2d(2)

    # takes input and style
    def forward(self, x, style):
        batch, channels, in_h, in_w = x.shape
        assert channels == self.fi

        # Modulate weight using style
        style = self.mod_fc(style).view(batch, 1, self.fi, 1, 1)
        scale = self.w_scale * style
        w = self.w * scale

        # Demodulate
        if self.do_demod:
            # In the paper the sum is over indices "i" and "k"
            # Representing out channel (index 2 in self.w)
            # and kernel size, which is two indices (3 and 4)
            demod_mult = torch.rsqrt(w.pow(2).sum([2,3,4]) + EPSILON).view(batch, self.fo, 1, 1, 1)
            w = w * demod_mult

        w = w.view(batch * self.fo, self.fi, self.k, self.k)
        x = x.view(1, batch * self.fi, in_h, in_w)

        x = conv2d_resample.conv2d_resample(x = x, w = w, f = self.fir, up = self.up, down = self.down,
                padding = self.pad, groups = batch, flip_weight = (self.up == 1))
        _, _, out_h, out_w = x.shape
        x = x.view(batch, self.fo, out_h, out_w)
        return x

# To RGB
class tRGB(nn.Module):
    def __init__(self, fi, dim_style):
        super().__init__()

        self.conv = modConv(fi, CHANNELS, 1, dim_style, do_demod = False)
        self.bias = nn.Parameter(torch.zeros(1, CHANNELS, 1, 1))
        
        self.up = UpsamplingBilinear2d(2)
        self.fir_k = torch.Tensor([1, 3, 3, 1]).cuda()
        self.fir_k = self.fir_k * 4 
        self.pad = (2, 1)

    def forward(self, x, style, y_last = None):
        y = self.conv(x, style) + self.bias
        if y_last is not None:
            #y_last = upfirdn2d.upsample2d(y_last, self.fir_k)
            y_last = self.up(y_last)

            y += y_last

        return y

# Single block in the generator
# i.e. modulated conv, noise, style, all put together
class modBlock(nn.Module):
    # As before mode can be "UP" "DOWN or None
    def __init__(self, fi, fo, k, dim_style, mode = None, do_demod = True):
        super().__init__()
        
        self.conv = modConv(fi, fo, k, dim_style, mode, do_demod)
        self.bias = nn.Parameter(torch.zeros(1, fo, 1, 1))
        self.noise_scale = nn.Linear(1, fo)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, style, noise_inject):
        x = self.conv(x, style)

        _, _, h, w = x.shape
        noise_inject = noise_inject[:,:h,:w,:] # [N, h, w, 1]
        noise_inject = self.noise_scale(noise_inject) # [N, h, w, fo]
        noise_inject = noise_inject.permute(0, 3, 2, 1) # [N, fo, h, w]
        x += noise_inject
        
        x = x + self.bias
        x = self.act(x)

        return x

# Larger block consisting of three modulated convs
class GenBlock(nn.Module):
    def __init__(self, fi, fo, k, dim_style):
        super().__init__()

        self.conv1 = modBlock(fi, fo, k, dim_style, mode = "UP")
        self.conv2 = modBlock(fo, fo, k, dim_style)
        self.to_rgb = tRGB(fo, dim_style)

    # Note, we expect 3 style vectors
    def forward(self, x, style, noise_inject, y_last = None):
        x = self.conv1(x, style[0], noise_inject)
        x = self.conv2(x, style[1], noise_inject)
        skip = self.to_rgb(x, style[2], y_last)

        return x, skip

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

# Minibatch standard deviation layer (from ProgGAN)
class MiniBatchSTD(nn.Module):
    def __init__(self, fi, fo):
        super().__init__()

        self.conv = discConv(fi + 1, fo, 3)

    def forward(self, x):
        n, c, h, w = x.shape
     
        minibatch_std = x.view(4, -1, 1, c, h, w)
        minibatch_std = minibatch_std.std(dim = 0, unbiased = False)
        minibatch_std = minibatch_std.mean([2, 3, 4], keepdims = True).squeeze(2)
        minibatch_std = minibatch_std.repeat(4, 1, h, w)
        x = torch.cat([x, minibatch_std], 1)

        return self.conv(x)
