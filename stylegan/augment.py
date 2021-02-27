import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as Ft
from torchvision.transforms import RandomAffine as RA
from torchvision.transforms import ColorJitter as CJ
from constants import *

# Augment layer that has a chance to apply augmentation
class AugmentLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # This covers all transformations
        # from the paper that can be reproduced with torchvision
        # transforms (tried to keep it simple)

        # Blit and geometric
        self.geometric = [
            xFlip,
            randRotate,
            intTranslate,
            RA(0, scale = (1, 1.5)), # random scaling between 1 and 1.5
            RA(180) # Random rotation between -180 and 180 degrees
        ]
        # Color
        self.color = [
            CJ([0, 7]),
            CJ(0, [0, 7]),
            CJ(0, hue = 0.5),
            CJ(0, saturation = [0, 7]) 
        ] 

        # For adaptive part
        self.aug_prob = 0
        self.updates = 0
        self.r_t = 0
        # Sum of all discriminator outputs and overall number of outputs
        # Used to get expectation (mean)
        self.d_outs = torch.Tensor([0])
        self.num_outs = 0

    def forward(self, x, prob = None):
        if prob is None: prob = self.aug_prob
        for augment in self.geometric:
            if randomPass(prob):
                x = augment(x)

        for augment in self.color:
            if randomPass(prob):
                x = augment(x)

        return x

    def adapt(self, real_labels):
        self.d_outs += torch.sign(real_labels).sum().item()
        self.num_outs += real_labels.shape[0]
        self.updates += 1

        if self.updates % ADA_INTERVAL == 0:
            self.r_t = self.d_outs / self.num_outs

            if self.r_t > AUG_P_TARGET:
                sign = 1
            else:
                sign = -1

            self.aug_prob += sign * self.num_outs / ADA_LENGTH
            self.aug_prob = min(self.aug_prob, 1)
            self.aug_prob = max(self.aug_prob, 0)

            self.d_outs *= 0
            self.num_outs = 0

# given a probability p
# returns True 100p% of the time
# False 100(1-p)% of the time
def randomPass(p):
    z = torch.rand(1).item()
    if z < p:
        return True
    return False

# Augmentations without CJ or RA
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
