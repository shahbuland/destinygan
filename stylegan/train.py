import torch
from torch import nn
from torchvision.datasets import ImageFolder

import util
import loss_funcs
from constants import *

def train(g, d, data_path):
    # Load dataset
    print("Loading dataset...")
    train_data = torch.utils.data.DataLoader(ImageFolder(data_path
