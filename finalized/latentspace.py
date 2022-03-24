import torch

from stylegan.models import Gen2
from stylegan.constants import *
import stylegan.util as util

import pygame
import numpy as np
import random
import os
import copy
import math

WEAPON_TYPE = "Random"

g = Gen2("./models/" + WEAPON_TYPE + "/", "gparams.pt")


(width, height) = (1024, 1024)
screen = pygame.display.set_mode((width, height))

u = torch.randn(LATENT_DIM).cuda()
v = torch.randn(LATENT_DIM).cuda()

ws_meter = 0
ad_meter = 0
qe_meter = 0

step_size = 0.0025

skip_mapping = True

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                ws_meter = 0
                ad_meter = 0
                u = torch.randn(LATENT_DIM, device = 'cuda')
                v = torch.randn(LATENT_DIM, device = 'cuda')
            if event.key == pygame.K_t:
                skip_mapping = not skip_mapping

    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        ws_meter += step_size
    if keys[pygame.K_s]:
        ws_meter -= step_size
    if keys[pygame.K_a]:
        ad_meter -= step_size
    if keys[pygame.K_d]:
        ad_meter += step_size
    if keys[pygame.K_q]:
        qe_meter -= step_size * 50
    if keys[pygame.K_e]:
        qe_meter += step_size * 50
    
    latent = torch.randn(LATENT_DIM).cuda() * ws_meter
    g.update_state_dict(qe_meter)
    
    print(latent.norm().item(), skip_mapping, qe_meter)


    sample = g.generate(latent, skip_mapping)

    sample = (255 * sample).permute(2, 1, 0).cpu().numpy().astype(np.uint8)


    surf = pygame.surfarray.make_surface(sample)
    screen.blit(surf, (0, 0))

    pygame.display.update()

pygame.quit()
