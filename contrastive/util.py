import torch
import cv2
import os

from constants import *

import numpy as np
import einops as eo
import asyncio
from copy import deepcopy

def get_dataset():
    ico_paths = os.listdir("./ico")
    gun_paths = os.listdir("./gun")
    ico_paths.sort()
    gun_paths.sort()

    ico_paths = ["./ico/" + path for path in ico_paths]
    gun_paths = ["./gun/" + path for path in gun_paths]

    n_data = len(ico_paths)
    ico = np.zeros((n_data, 128, 128, 3))
    gun = np.zeros((n_data, 1024, 1024, 3))

    for i in range(n_data):
        ico_path = ico_paths[i]
        gun_path = gun_paths[i]

        def get(path, size):
            im = cv2.imread(path)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            return cv2.resize(im, (size, size),
                        interpolation = cv2.INTER_CUBIC)

        ico[i] = get(ico_path, 128)
        gun[i] = get(gun_path, 1024)

    ico = torch.from_numpy(ico).permute(0, 3, 1, 2)
    gun = torch.from_numpy(gun).permute(0, 3, 1, 2)

    return (gun, ico)

# Pytorch data loaders are terrible on windows so I wrote my own
# Expects constants batch size
class DataLoader:
    def __init__(self, bs):
        ico_paths = os.listdir("./ico")
        gun_paths = os.listdir("./gun")
        ico_paths.sort()
        gun_paths.sort()

        self.ico_paths = ["./ico/" + path for path in ico_paths]
        self.gun_paths = ["./gun/" + path for path in gun_paths]

        self.n_data = len(ico_paths)
        
        self.bs = bs
        self.data_buffer = [torch.empty(BATCH_SIZE, 3, 1024, 1024),
                        torch.empty(BATCH_SIZE, 3, 128, 128)]

        loop = asyncio.get_event_loop()

        self.populate_buffer()

    def get(self, path, size):
        im = cv2.imread(path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (size, size),
                        interpolation = cv2.INTER_CUBIC)
        im = eo.rearrange(im, 'h w c -> c h w')
        im = (torch.from_numpy(im) - 127.5)/127.5
        return im

    def populate_buffer(self):
        batch_inds = torch.randint(self.n_data, (self.bs,))

        for i, ind in enumerate(batch_inds):
            self.data_buffer[0][i] = self.get(self.gun_paths[ind], 1024)
            self.data_buffer[1][i] = self.get(self.ico_paths[ind], 128)
    
    def next(self):
        ret = deepcopy(self.data_buffer)
        self.populate_buffer()
        return ret
            
def get_toy():
    return (torch.zeros(900, 3, 1024, 1024),
            torch.zeros(900, 3, 128, 128))
