import torch
from torch import nn
from torch.autograd import grad
import torch.nn.functional as F
import math

from constants import *
import util

from torch_utils.ops import conv2d_gradfix

# Perpetual path length
def pathLength(latents, batch):
    n, c, h, w = batch.shape
    n_pixels = h * w
    pl_noise = torch.randn_like(batch) / math.sqrt(n_pixels)
    outs = (batch * pl_noise).sum()

    gradients = grad(outputs = outs,
                    inputs = latents,
                    grad_outputs = torch.ones_like(outs), 
                    create_graph = True, retain_graph = True,
                    only_inputs = True)[0]
    return (gradients ** 2).sum(dim = 2).mean(dim = 1).sqrt()


# Logistic loss for discriminator
def discLoss(real_labels, gen_labels):
    real_loss = F.softplus(-1 * real_labels) # Maximize real labels
    gen_loss = F.softplus(gen_labels) # Minimize fake labels
    return real_loss.mean() + gen_loss.mean()

# For generator
def genLoss(gen_labels):
    return F.softplus(-1 * gen_labels).mean() # Maximize fake labels

# R1 regularization term for discriminator
def r1Reg(batch, labels):
    n, _, _, _ = batch.shape
    gradients = grad(outputs = labels,
                                    inputs = batch,
                                    grad_outputs = torch.ones_like(labels),
                                    create_graph = True,
                                    retain_graph = True, only_inputs = True)[0]
    gradients = util.flatten(gradients)
    return ((gradients.norm(2, dim = 1) - 1)**2).mean()
