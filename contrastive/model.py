import torch
from torch import nn
import torch.nn.functional as F

import layers
from constants import *

import math
import numpy as np

# This is hard coded for 128x128 icons and 1024x1024 guns

class ImageEncoder(nn.Module):
    def __init__(self, size):
        super().__init__()

        ch = [32, 64, 128, 256, 512, 512, 512]
        base_size = 128
        while base_size < size:
            ch.append(512)
            base_size *= 2

        self.convLayers = nn.Sequential(*[
                                        layers.discConv(3, ch[0], 1)] + \
                                        [layers.DiscBlock(ch[i], ch[i+1]) for i
                                        in range(len(ch) - 2)])
        
        self.denseLayers = nn.Sequential(*[
                    layers.modLinear(ch[-1] * 4 * 4, ch[-1], use_act = True),
                    layers.modLinear(ch[-1], LATENT_DIM)])

    def forward(self, x):
        x = self.convLayers(x)
        x = x.view(-1, 512 * 4 * 4)
        return self.denseLayers(x)

class ContrastiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.ico_enc = ImageEncoder(128)
        self.gun_enc = ImageEncoder(1024)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.clamp_min = np.log(1/100)
        self.clamp_max = np.log(100)

    def clamp(self):
        with torch.no_grad():
            self.logit_scale.clamp(self.clamp_min, self.clamp_max)

    def encodeIco(self, x):
        return self.ico_enc(x)

    def encodeGun(self, x):
        return self.gun_enc(x)

    # Gets contrastive loss between two embeddings
    def cLoss(self, x, y):
        n = x.shape[0]

        x = F.normalize(x)
        y = F.normalize(y)
        
        logits = x @ y.T * self.logit_scale.exp()
        labels = torch.arange(n, device = 'cuda')

        loss_ico = F.cross_entropy(logits, labels)
        loss_gun = F.cross_entropy(logits.T, labels)

        acc_ico = (torch.argmax(logits, dim = 1) == labels).sum()
        acc_gun = (torch.argmax(logits, dim = 0) == labels).sum()

        return (loss_ico + loss_gun) / 2, (acc_ico + acc_gun) / n / 2
