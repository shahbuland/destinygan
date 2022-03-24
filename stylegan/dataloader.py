import torch
import numpy as np
import cv2
import os

from constants import *

# Given path to dataset, makes dataloader
# Doesn't perform any checks on dataset, so assumes
# format is correct (datasets/[dataset_name]/[dataset_name]_png/...
class MyDataLoader: 
    def __init__(self, dataset_name, size = BATCH_SIZE):
        root = "./datasets/" + dataset_name + "/" + dataset_name + "_png/"
        paths = [root + path for path in os.listdir(root)]

        self.size = size
        self.num_samples = len(paths)
        self.dataset = np.zeros((self.num_samples, IMG_SIZE, IMG_SIZE, 3))
        
        for i, path in enumerate(paths):
            im = cv2.imread(path)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_CUBIC)
            self.dataset[i] = im

        self.dataset = torch.from_numpy(self.dataset)
        self.dataset = self.dataset.permute(0, 3, 1, 2) # NCHW

    def next(self):
        inds = torch.randint(self.num_samples, (self.size,), dtype = torch.int64)
        batch = self.dataset[inds].to(DEVICE).float()
        batch = (batch - 127.5) / (127.5) # [-1, 1]
        return batch
