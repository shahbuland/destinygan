import torch
from torch import nn
from torch.nn import functional as F

def upsample(x):
    return F.interpolate(x, (2,2))
