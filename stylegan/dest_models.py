import torch
from torch import nn

from constants import *
import layers
import wavelets
import util
import einops as eo

# This script has modified generator and discriminator 
# that are very domain specific 
# - SWAGAN architecture
# - Specific to 1024x1024 and 128x128
# - Both D and G work on two images at once with one latent
# - Purpose is to generate small ui icon and gun at same time

def DoubledGenerator(nn.Module):
    def __init__(self, dim_style = LATENT_DIM, mapperDepth = N_MLP):
        super().__init__()

        self.dim_style = dim_style
        self.mappingNetwork = []
        self.mappingNetwork.append(layers.mappingNorm())
        for _ in range(mapperDepth):
            self.mappingNetwork.append(
                layers.modLinear(dim_style,
                                 dim_style, lr_mult = 1e-2, use_act = True))

        self.gunConst = layers.constantIn(512, 4)
        self.gunFirstConv = layers.modBlock(512, 512, 3, dim_style, mode = None)
        self.gunFirstRGB = wavelets.tRGB(512, dim_style)

        self.icoConst = layers.constantIn(512, 4)
        self.icoFirstConv = layers.modBlock(512, 512, 3, dim_style, mode = None)
        self.icoFirstRGB = wavelets.tRGB(512, dim_style)

        ch_ico = [512, 256, 128, 64, 32]
        ch_gun = [512, 512, 512] + ch_ico
        self.n_layers_ico = len(ch_ico) - 1
        self.n_layers_gun = len(ch_gun) - 1
        
        GenBlock = wavelets.GenBlock
        self.gunConvLayers = nn.Sequential(*[GenBlock(ch_gun[i], ch_gun[i+1], 3, dim_style)
                                            for i in range(self.n_layers_gun)])
        self.icoConvLayers = nn.Sequential(*[GenBlock(ch_ico[i], ch_ico[i+1], 3, dim_style)
                                            for i in range(self.n_layers_ico)])
        self.upWT = wavelets.WaveletTransform(inverse = True)

    def get_w_bar(self, z):
        return torch.mean(self.mappingNetwork(z), dim = 0)

    def forward(self, z, w_bar = None, mix_index = None):
        
        is_mix = len(list(z.shape)) > 2
        if is_mix:
            _, n, _ = z.shape
        else:
            n, _ = z.shape
            z = [z]

        w = [self.mappingNetwork(z_i) for z_i in z]

        # Truncation
        if w_bar is not None:
            for i, _ in enumerate(w):
                w[i] = w_bar + trunc_psi * (w[i] - w_bar)

        # Style mixing
        # This implementation effectively decreases mixing reg prob for ico gen,
        # but I felt this was better than giving them different mix indices
        n_latent = 3*(self.n_layers_gun + 1) # This many injection points for style in gun conv
        latent = eo.repeat(w[0], 'b d -> l b d', l = n_latent)
        if is_mix:
            mix_latent = w[1]
            if mix_index is None:
                mix_index = np.random.randint(0, n_latent)
            latent[mix_index:] = latent2

        noise_injection = torch.empty(n, IMG_SIZE, IMG_SIZE, 1, device = 'cuda').uniform(0.,1.)

        # Make icon first
        ico = self.icoConst(latent[0])
        ico = self.icoFirstConv(ico, latent[1], noise_injection)
        ico_res = self.icoFirstRGB(ico, latent[2]) # res as in "residue"

        for ind, layer in enumerate(self.icoConvLayers):
            
