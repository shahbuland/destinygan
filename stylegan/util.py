import torch
import cv2
import torch.nn.functional as F
import numpy as np

from constants import *

# Freezing and unfreezing weights in model
def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze_model(model):
    for p in model.paramters():
        p.requires_grad = True

# Flatten a tensor, accounting for batch axis
def flatten(t):
    return t.reshape(t.shape[0], -1)

# Convert a batch of PIL images into a tensor
def imageToTensor(img):
    t = torch.from_numpy(img).float()
    t = t.permute(0, 3, 1, 2) / 255
    if USE_CUDA: t = t.cuda()
    if USE_HALF: t = t.half()
    t = F.interpolate(t, (IMG_SIZE, IMG_SIZE))
    return t

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
