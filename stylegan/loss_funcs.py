import torch
from torch import nn
from torch import autograd
import torch.nn.functional as F

from constants import *
import util

# Perpetual path length
# Takes input output pair from generator
# And a moving average of path length
def PPL(latents, gen_img, avg_path_len):
    n, c, h, w = gen_img.shape
    y = torch.randn_like(gen_img) / ((h * w)**.5)

    # Get jacobian * random image wrt input latent vector
    grad, = autograd.grad((gen_img * y).sum(), latents, create_graph = True)
    # L2 norm of jacobian, when this is 0, jacobian orthogonal
    l2norm = torch.sqrt(grad.pow(2).sum(2).mean(1))
    # TODO: Do you really need sqrt for l2 norm as a loss term? check computation cost

    a = avg_path_len + PPL_DECAY * (l2norm.mean() - avg_path_len)

    # E[(||J^Ty|| - a)^2]
    loss = (l2norm - a).pow(2).mean()

    # Return last moving average for next calculation
    return loss, l2norm, a

# Logistic loss for discriminator
def discLoss(real_labels, gen_labels):
    real_loss = F.softplus(-1 * real_labels) # Maximize real labels
    gen_loss = F.softplus(gen_labels) # Minimize fake labels
    return real_loss.mean() + gen_loss.mean()

# For generator
def genLoss(gen_labels):
    return F.softplus(-1 * gen_labels).mean() # Maximize fake labels

# R1 regularization term for discriminator
def r1Reg(real_imgs, real_labels):
    out = real_labels.sum()
    grad, = autograd.grad(out, real_imgs, create_graph = True)
    loss = util.flatten(grad.pow(2)).sum(1).mean() * R1REG_GAMMA / 2
    return loss
