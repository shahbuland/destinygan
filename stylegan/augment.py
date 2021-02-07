import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as Ft

from constants import *

# Augment layer that has a chance to apply augmentation
class AugmentLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = torch.rand(1)
        if z.item() > AUG_P:
            return x

        # Following code runs with probability AUG_P

# All augmentations from stylegan2 ada paper
# All assume N x C x H x W input

def xFlip(t):
    return t.flip([3])     

def randRotate(t, rot = None):
    z = torch.randint(3, (1,)).item()
    if rot is not None: z = rot # For debug
    if z == 0: # 90 deg
        # Swap cols and rows, then flip rows
        return t.permute(0, 1, 3, 2).flip([2])
    elif z == 1: # 180 deg
        # flip rows and cols
        return t.flip([2, 3])
    else: # 270 deg
        # Swap rows and cols, then flip cols
        return t.permute(0, 1, 3, 2).flip([3])

def intTranslate(t):
    _, _, h, w = t.shape
    x_translate, y_translate = torch.rand(2) * 0.125 * 2 - 0.125
    x_translate = int(x_translate.item() * w)
    y_translate = int(y_translate.item() * h)
    # Do y translation first
    t = torch.cat([t[:,:,y_translate:h,:], t[:,:,0:y_translate,:]], axis = 2)
    # Then x
    t = torch.cat([t[:,:,:,x_translate:w], t[:,:,:,0:x_translate]], axis = 3)
    return t

def isoTropScale(
