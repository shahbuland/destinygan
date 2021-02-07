import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from constants import *
import ops

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
        y = self.inp_noise.repeat(n, 1, 1, 1)
        return y

# Modulated convolution
class modConv(nn.Module):
    # Mode can be "UP", "DOWN" or None
    def __init__(self, fi, fo, k, dim_style, mode = None, do_demod = True):
        super().__init__()

        self.fi = fi # Filters/channels in
        self.fo = fo # Filters/channels out
        self.k = k # Kernel size
        self.mode = mode 
        self.do_demod = do_demod

        self.pad = k // 2
        
        self.w = nn.Parameter(torch.randn(1, fo, fi, k, k))# Weight matrix
        self.w_scale = (fi * (k ** 2))**-.5
        # Maps from style latent space to fi
        self.mod_fc = modLinear(dim_style, fi)
        
    # takes input and style
    def forward(self, x, style):
        batch, channels, in_h, in_w = input.shape
        assert channels == self.fi

        # Modulate weight using style
        style = self.mod_fc(style).view(batch, 1, channels, 1, 1)
        w = self.w * self.w_scale * style

        # Demodulate
        if self.do_demod:
            # In the paper the sum is over indices "i" and "k"
            # Representing out channel (index 2 in self.w)
            # and kernel size, which is two indices (3 and 4)
            demod = weight * torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + EPSILON)

        w = w.view(batch * self.fo, self.fi, self.k, self.k)

        x = x.view(1, batch * self.fi, in_h, in_w)

        if mode == "DOWN":
            y = F.conv2d(x, w, padding = 0, stride = 2, groups = batch)
        elif mode == "UP":
            # For transposed convolution need filters out and in
            # swapped
            w = w.view(batch, self.fo, self.fi, self.k, self.k)
            w = w.transpose(1, 2)
            w = w.view(batch * self.fi, self.fo, self.k, self.k)

            y = F.conv_transpose2d(x, w, padding = 0, stride = 2, groups = batch)

        else:
            y = F.conv2d(x, w, padding = self.pad, groups = batch)

        _, _, out_h, out_w = y.shape
        y = y.view(batch, self.fo, out_h, out_w)

        return y

# To RGB
class tRGB(nn.Module):
    def __init__(self, fi, dim_style, up = True):
        super().__init__()

        self.conv = modConv(fi, CHANNELS, 1, dim_style, do_demod = False)
        self.bias = nn.Parameter(torch.zeros(1, CHANNELS, 1, 1))

    # y_last is for progressive growing
    def forward(self, x, style, y_last = None):
        y = self.conv(x, style) + self.bias

        if y_last is not None:
            y_last = ops.upsample(y_last)
            y += y_last

        return y

# Layer that adds noise
class Noise(nn.Module):
    def __init__(self):
        super().__init__()

        # "Learned per channel scaling factor" for noise
        self.w = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        n, _, h, w = x.shape
        noise = torch.randn(n, 1, h, w)

        return x + w * noise

# Single block in the generator
# i.e. modulated conv, noise, style, all put together
class modBlock(nn.Module):
    # As before mode can be "UP" "DOWN or None
    def __init__(self, fi, fo, k, dim_style, mode = None, do_demod = True):
        super().__init__()
        
        self.conv = modConv(fi, fo, k, dim_style, mode, do_demod)
        self.noise = Noise()
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, style):
        x = self.conv(x, style)
        x = self.noise(x)
        x = self.act(x)

        return x

# Larger block consisting of three modulated convs
class GenBlock(nn.Module):
    def __init__(self, fi, fo, k, dim_style):
        super().__init__()

        self.conv1 = modBlock(fi, fo, k, dim_style, mode = "UP")
        self.conv2 = modBlock(fo, fo, k, dim_style)
        self.to_rgb = tRGB(fo, dim_style)

    def forward(self, x, style, y_last = None):
        x = self.conv1(x, style)
        x = self.conv2(x, style)
        skip = self.to_rgb(x, style, y_last)

        return x, skip

# Conv layers for the discriminator
class discConv(nn.Module):
    # Mode can be "DOWN" or None
    def __init__(self, fi, fo, k, mode = None, use_bias = True, use_act = True):
        super().__init__()

        self.w = nn.Parameter(torch.randn(fo, fi, k, k))
        self.w_scale = (fi * k * k)**-.5
        
        self.b = nn.Parameter(torch.zeros(fo)) if use_bias else None
        self.act = nn.LeakyReLU(0.2) if use_act else None

        self.stride = 2 if mode == "DOWN" else 1
        self.pad = 0 if mode == "DOWN" else k // 2

    def forward(self, x):
        x = F.conv2d(x, self.w * self.w_scale, self.b, self.stride, self.pad)

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
