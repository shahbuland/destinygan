import torch
import cv2
import torch.nn.functional as F
from torch import nn
import numpy as np
import os

from constants import *

# ==== MODEL SPECIFIC ====

# Freezing and unfreezing weights in model
def freezeModel(model):
    for p in model.parameters():
        p.requires_grad = False

def unfreezeModel(model):
    for p in model.parameters():
        p.requires_grad = True

# Flatten a tensor, accounting for batch axis
def flatten(t):
    return t.reshape(t.shape[0], -1)

# Normalize FIR kernel
def norm_fir_k(k):
    k = k[None, :] * k[:, None]
    k = k / k.sum()
    return k

# ==== IMAGE PROCESSING ====

def normalizeImTensor(img):
    flat = img.view(-1)
    img -= flat.min()
    img /= flat.max()
    return img

# Convert a batch of PIL images into a tensor
def imageToTensor(img):
    t = torch.from_numpy(img).float()
    t = t.permute(0, 3, 1, 2) / 255
    t = F.interpolate(t, (IMG_SIZE, IMG_SIZE))
    return t

from torchvision import transforms
# Same as above but for torchvision transform
def getImToTensorTF():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace = True),
        transforms.Resize((IMG_SIZE, IMG_SIZE))])

# Convert a single CV image into a tensor
def imageToTensorCV(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return imageToTensor(np.expand_dims(img, 0))

# Convert a tensor into an image and save to disk at desired path
def tensorToImage(t, path):
    # Assumes C x H x W input
    t = (t.permute(1, 2, 0) * 255)

# Convert to cv image
def tensorToCV(t):
    t = t.cpu().detach().permute(1, 2, 0)
    t = t.numpy()
    t = cv2.cvtColor(t, cv2.COLOR_RGB2BGR)
    return t

# ==== TRAINING ====

# Returns 2 noise vectors with prob MIX_PROB
# 1 otherwise (latent vectors)
def getMixingLatent(size, device = DEVICE):
    if np.random.random() < MIX_PROB:
        return torch.randn(2, size, LATENT_DIM, device = DEVICE)
    return torch.randn(size, LATENT_DIM, device = DEVICE)

import torchvision
# N x C x H x W input on [-1, 1]
def drawSamples(samples, path):
    im = torchvision.utils.make_grid(samples, nrow = 8, normalize = True, range = (-1, 1))
    torchvision.utils.save_image(im, path)

# exponential moving average for model parameters
def get_ema(ema_model, new_model, alpha = EMA_ALPHA):
    old_params = dict(ema_model.named_parameters())
    new_params = dict(new_model.named_parameters())

    # S_t = (1 - alpha) * S_{t-1} + alpha * new_model
    # Where S_{t-1} will just be the ema_model
    for key in old_params.keys():
        old_params[key].data.mul_(1 - alpha).add_(new_params[key], alpha = alpha)

# === GENERAL ===

# Quickly make vector filled with one value
def fillVec(size, val, device = DEVICE):
    return val * torch.ones(size, device = DEVICE)

# path to folder and some base string y,
# gets all paths of form root/Xy.pt, where X is numbers
# Sorts paths by X
def get_paths(root, y):
    paths = os.listdir(root)
    def key(path):
        first_y = path.find(y)
        if(first_y == -1): return -1
        return int(path[:first_y])
    
    paths = sorted(paths, key=key)
    # Remove those that didn't have y at all
    i = 0
    for path in paths:
        if(key(path) == -1):
            i += 1
        else:
            break
    paths = paths[i:]

    return [root + path for path in paths]

import copy
# Interpolate between two state dicts
def lerp_state_dict(a, b, t):
    if(t <= 0.0):
        return a
    if(t >= 1.0):
        return b

    c = copy.deepcopy(a)
    
    for key in a.keys():
        # c = a + t * (b - a)
        c[key].data.lerp_(b[key], t)

    return c

# Initialize generator
def init_g_weights(g):
    for m in g.modules():
            if type(m) == nn.Conv2d or type(m) == nn.Linear:
                nn.init.kaiming_normal_(m.weight)
    nn.init.zeros_(g.firstConv.noise_scale.weight)
    nn.init.zeros_(g.firstConv.noise_scale.bias)
    for layer in g.convLayers:
        nn.init.zeros_(layer.conv1.noise_scale.weight)
        nn.init.zeros_(layer.conv2.noise_scale.weight)
        nn.init.zeros_(layer.conv1.noise_scale.bias)
        nn.init.zeros_(layer.conv2.noise_scale.bias)

# Initialize discriminator
def init_d_weights(d):
    for m in d.modules():
            if type(m) == nn.Conv2d or type(m) == nn.Linear:
                nn.init.kaiming_normal_(m.weight)
