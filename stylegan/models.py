import torch
from torch import nn
import torch.nn.functional as F
from constants import *
import layers

class Generator(nn.Module):
    def __init__(self, dim_style, mapperDepth, channel_mul = 2):
        super().__init__()
    
        self.dim_style = dim_style

        self.mappingNetwork = nn.ModuleList()

        self.mappingNetwork.append(layers.mappingNorm())
        for _ in range(mapperDepth):
            self.mappingNetwork.append(layers.modLinear(dim_style, dim_style, lr_mult = 0.01, use_act = True))

        self.constIn =  layers.constantIn(512, 4)
        self.firstConv = layers.modBlock(512, 512, 3, dim_style, mode = None)
        self.firstRGB = layers.tRGB(512, dim_style, up = False)

        ch = [512, 512, 512, 512, 512, 256, 128, 64, 32]

        self.convLayers = nn.ModuleList()

        for i in range(len(ch) - 1):
            self.convLayers.append(layers.GenBlock(ch[i], ch[i + 1], 3, dim_style))

        def get_w_bar(self, z):
            return torch.mean(self.mappingNetwork(z), dim = 0)

        # For the sake of following the paper, 
        # I use variable names from StyleGAN paper,
        # So refer to the paper if a variable name is vague
        def forward(self, z, skip_mapping = False, w_bar = None):
            if not skip_mapping:
                w = self.mappingNetwork(z)

            # Using truncation trick
            w = w_bar + TRUNC_PSI * (w - w_bar)

            # Should now be able to get 2 latents (y_s and y_b from paper)
            latent = w

            # Infers batch shape from latent vector, doesn't actually use it
            out = self.constIn(latent)
            out = self.firstConv(out, latent)
            y_last = self.firstRGB(out)

            for layer in self.ConvLayers:
                out, y_last = layer(out, y_last)

            y = y_last

            return y, latents


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        ch = [32, 64, 128, 256, 512, 512, 512, 512, 512]

        self.convLayers = nn.ModuleList()
        
        self.firstConv = layers.discConv(3, ch[0], 1) # From RGB
        for i in range(len(ch) - 2):
            self.convLayers.append(layers.DiscBlock(ch[i], ch[i + 1]))
        self.lastConv = layers.discConv(ch[-1] + 1, ch[-1], 3)

        self.fc1 = layers.modLinear(ch[-1] * 4 * 4, ch[-1], use_act = True)
        self.fc2 = layers.modLinear(ch[-1], 1)

    def forward(self, x):
        x = self.firstConv(x)
        for layer in self.convLayers:
            x = self.layer(x)
        
        n, c, h, w = x.shape

        # Minibatch standard deviation (from ProgGAN)
        minibatch = x.view(4, -1, self.feature_std, c, h, w)
        minibatch_std = minibatch.std(dim = 0, unbiased = False)
        minibatch_std = minibatch_std.mean([2, 3, 4], keepdims = True).squeeze(2)
        minibatch_std = minibatch_std.repeat(4, 1, h, w)
        x = torch.cat([x, minibatch_std], 1)

        x = self.lastConv(x)
        x = x.view(n, -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


