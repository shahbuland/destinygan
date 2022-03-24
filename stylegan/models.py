import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from constants import *
import layers
import wavelets
import util
import wandb

# Latent to w space
class MappingNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(*[
            layers.modLinear(LATENT_DIM, LATENT_DIM, lr_mult = 1e-2, use_act = True)
            for _ in range(N_MLP)])

    def forward(self, x):
        x = F.normalize(x, dim = 1)
        return self.layers(x)

class Generator(nn.Module):
    def __init__(self, size = IMG_SIZE, dim_style = LATENT_DIM):
        super().__init__()
    
        self.dim_style = dim_style

        self.mappingNetwork = MappingNetwork()

        if SIMPLIFY_GEN:
            self.constIn = layers.simplifyIn(512, 4)
        else:
            self.constIn =  layers.constantIn(512, 4)
        self.firstConv = layers.modBlock(512, 512, 3, dim_style, mode = None)
        
        if SWAGAN:
            self.firstRGB = wavelets.tRGB(512, dim_style)
        else:
            self.firstRGB = layers.tRGB(512, dim_style)

        ch = [512, 256, 128, 64, 32]
        if not SWAGAN: ch = [512] + ch
        # base img size with above list would be 128
        base_size = 128
        while base_size < IMG_SIZE:
            base_size *= 2
            ch = [512] + ch

        self.convLayers = nn.ModuleList()
        self.num_layers = len(ch) - 1

        if SWAGAN:
            GenBlock = wavelets.GenBlock
        else:
            GenBlock = layers.GenBlock

        for i in range(self.num_layers):
            self.convLayers.append(GenBlock(ch[i], ch[i + 1], 3, dim_style))

        if SWAGAN:
            self.upWT = wavelets.WaveletTransform(inverse = True)

    def get_w_bar(self, z):
        return torch.mean(self.mappingNetwork(z), dim = 0)
    
    # For the sake of following the paper, 
    # I use variable names from StyleGAN paper,
    # So refer to the paper if a variable name is vague
    # Mix index (for style mixing) is index at which we
    # introduce second style
    def forward(self, z, skip_mapping = SKIP_MAPPING, w_bar = None, mix_index = None, trunc_psi = TRUNC_PSI):
        
        # Use multiple styles instead of 1
        is_mix = len(list(z.shape)) > 2
        if is_mix:
            _, n, _ = z.shape
        else:
            n, _ = z.shape

        # Mapping
        if not skip_mapping:
            if is_mix:
                w = [self.mappingNetwork(z_i) for z_i in z]
            else:
                w = [self.mappingNetwork(z)]
        else:
            if is_mix: w = z
            else: w = [z]

        # Using truncation trick
        if w_bar is not None:
            for i, _ in enumerate(w):
                w[i] = w_bar + trunc_psi * (w[i] - w_bar)


        # Style mixing
        latent1 = w[0]
        n_latent = 3*(self.num_layers + 1) # For every block, 3 places to input style
        latent = torch.cat([latent1.unsqueeze(0) for _ in range(n_latent)])
        if is_mix:
            latent2 = w[1]
            if mix_index is None:
                mix_index = np.random.randint(0, n_latent)
            latent[mix_index:] = latent2

        # Noise injection
        noise_injection = torch.empty(n, IMG_SIZE, IMG_SIZE, 1, device = 'cuda').uniform_(0.,1.)

        # Infers batch shape from latent vector, doesn't actually use it
        out = self.constIn(latent[0])
        out = self.firstConv(out, latent[1], noise_injection)
        y_last = self.firstRGB(out, latent[2])
        
        for ind, layer in enumerate(self.convLayers):
            out, y_last = layer(out, latent[ind + 3:ind + 6], noise_injection,
                                y_last)
        
        y = y_last
        if SWAGAN:
            y = self.upWT(y)
        return y, latent

    # Mean latent vector for truncation trick
    def get_w_bar(self, n_samples):
        z = torch.randn(n_samples, LATENT_DIM, device = 'cuda')
        w = self.mappingNetwork(z)
        w_bar = w.mean(0, keepdim = True)
        return w_bar

    # Generate novel samples
    def generate(self, n_samples, latent = None, skip_mapping = False, trunc_psi = TRUNC_PSI):
        w_bar = self.get_w_bar(TRUNC_SAMPLES) if trunc_psi > 0 else None
        if latent is None:
            z = torch.randn(n_samples, LATENT_DIM, device = 'cuda')
        else:
            z = latent
        samples, _ = self.forward(z, w_bar = w_bar, skip_mapping = skip_mapping, trunc_psi = trunc_psi)
        return samples
        
class Discriminator(nn.Module):
    def __init__(self, size = IMG_SIZE):
        super().__init__()

        ch = [32, 64, 128, 256, 512, 512, 512]
        base_size = 128
        while base_size < size:
            base_size *= 2
            ch.append(512)

        self.convLayers = nn.ModuleList()

        self.firstConv = layers.discConv(3, ch[0], 1) # From RGB

        for i in range(len(ch) - 2):
            self.convLayers.append(layers.DiscBlock(ch[i], ch[i + 1]))
        
        # Minibatch standard deviation
        self.mb_sd = layers.MiniBatchSTD(ch[-1], ch[-1])

        self.fc1 = layers.modLinear(ch[-1] * 4 * 4, ch[-1], use_act = True)
        self.fc2 = layers.modLinear(ch[-1], 1)

    def forward(self, x):
        x = self.firstConv(x)
        for layer in self.convLayers:
            x = layer(x)
       
        n, _, _, _ = x.shape
        x = self.mb_sd(x)
        x = x.view(n, -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

# Discriminator for SWAGAN
class SwagDiscriminator(nn.Module):
    def __init__(self, size = IMG_SIZE):
        super().__init__()

        ch = [32, 64, 128, 256, 512, 512]
        base_size = 128
        while base_size < size:
            base_size *= 2
            ch.append(512)

        self.convLayers = nn.ModuleList()
        self.downWT = wavelets.WaveletTransform(inverse = False)

        for i in range(len(ch) - 2):
            self.convLayers.append(wavelets.DiscBlock(ch[i], ch[i + 1]))
        self.final_rgb = wavelets.fRGB(ch[-1])

        self.mb_sd = layers.MiniBatchSTD(ch[-1], ch[-1])
        self.fc1 = layers.modLinear(ch[-1] * 4 * 4, ch[-1], use_act = True)
        self.fc2 = layers.modLinear(ch[-1], 1)

    def forward(self, x, return_sd = False):
        x = self.downWT(x)

        skip = None
        for layer in self.convLayers:
            x, skip = layer(x, skip)
        _, x = self.final_rgb(x, skip)

        n, _, _, _ = x.shape
        mbsd = self.mb_sd(x)
        mbsd = mbsd.view(n, -1)
        x = self.fc1(mbsd)
        x = self.fc2(x)
        if not return_sd: return x
        return x, torch.norm(mbsd, dim = 1).sum()

import math

# Wrapper class for Generator that's modified for single image generation
# And handling intermediary models
class Gen2:
    def __init__(self, root, y):
        self.model = Generator()
        self.model.cuda()
        self.model.eval()

        # Keep all parameters in memory
        self.state_dict_paths = util.get_paths(root, y)
        self.num_models = len(self.state_dict_paths)

        if(self.num_models < 2):
            print("Error: expect at least 2 models")
            exit()
        
        self.lower = 0
        self.lower_params = torch.load(self.state_dict_paths[self.lower])
        self.upper_params = torch.load(self.state_dict_paths[self.lower + 1])
        
        self.model.load_state_dict(self.lower_params)

        self.t_past = 0

    # t in [0, self.num_models - 1]
    def update_state_dict(self, t):
        t = max(t, 0.0)
        t = min(t, 1.0 * (self.num_models - 1))

        if t == self.t_past: return
        else: self.t_past = t

        lower = math.floor(t)
        lerp_t = t - lower

        # Update param holders if needed
        if lower != self.lower:
            self.lower_params = torch.load(self.state_dict_paths[lower])
            self.upper_params = torch.load(self.state_dict_paths[lower + 1])
            self.lower = lower
            
        # Set to new t
        self.model.load_state_dict(util.lerp_state_dict(
            self.lower_params,
            self.upper_params,
            lerp_t))

    # Generate single image given style vectors (N x LATENT_DIM)
    def generate(self, style, skip_mapping = False):
        if(len(style.shape) == 1): style = style.unsqueeze(0)
        with torch.no_grad():
            sample = self.model.generate(1, latent = style,
                    skip_mapping = skip_mapping, trunc_psi = 0.2)
        
        sample = sample.squeeze()
        sample = self.normalize(sample)

        return sample

    def normalize(self, sample):
        # Normalize by (-1, 1) as (min, max)
        sample += 1
        sample /= 2
        return sample.clip(0, 1)
