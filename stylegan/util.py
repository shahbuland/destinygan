import torch
import cv2
import torch.nn.functional as F
import numpy as np

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

# Convert a batch of PIL images into a tensor
def imageToTensor(img):
    t = torch.from_numpy(img).float()
    t = t.permute(0, 3, 1, 2) / 255
    t = F.interpolate(t, (IMG_SIZE, IMG_SIZE))
    return t

from torchvision import transforms
# Same as above but for torchvision transform
imageToTensorTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMG_SIZE),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace = True)])

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
    t = t.cpu().detach().permute(0, 2, 3, 1)
    t = (255 * t.squeeze()).type(torch.ByteTensor).numpy()
    t = cv2.cvtColor(t, cv2.COLOR_RGB2BGR)
    return t

# ==== TRAINING ====

# Returns 2 noise vectors with prob MIX_PROB
# 1 otherwise (latent vectors)
def getMixingLatent(size):
    if np.random.random() < MIX_PROB:
        return torch.randn(2, size, LATENT_DIM)
    return torch.randn(size, LATENT_DIM)

import torchvision
# N x C x H x W input on [-1, 1]
def drawSamples(samples, path):
    im = torchvision.utils.make_grid(samples, nrow = 8, normalize = True, range = (-1, 1))
    torchvision.utils.save_image(im, path)

# === GENERATING ===


