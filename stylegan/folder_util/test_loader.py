import torch
from torchvision.datasets import ImageFolder
import torchvision
from torch import nn
import util

import numpy as np
import cv2
import os

from constants import *
from clock import Clock
from dataloader import MyDataLoader

# Script to test if dataloader is working as expected

if __name__ == '__main__':
    timer = Clock()

    timer.hit()
    mode = "MINE"
    if mode == "MINE":
        train_data = MyDataLoader("destgun")
    else:
        dataset = ImageFolder("./datasets/destgun/", transform = util.getImToTensorTF())
        train_data = torch.utils.data.DataLoader(dataset, batch_size = BATCH_SIZE,
                shuffle = True, pin_memory = True, num_workers = 0)

    print(timer.hit())

    TEST_STEPS = 10
    timer.hit()
    for ITER in range(TEST_STEPS):
        real = train_data.next() if mode == "MINE" else next(iter(train_data))[0]
        util.drawSamples(real, "./datatest.png")
        print(real.mean(), timer.hit())
