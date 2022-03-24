import torch
from torchvision.transforms import ToPILImage
import numpy as np

import random
import os
from PIL import Image, ImageFont, ImageDraw

from stylegan.models import Gen2
from stylegan.constants import *
import stylegan.util as util

from textgeneration import get_item_txt

# Take PIL Image, name, type, flavor test
# Overlays text in Destiny format
# Resizes image to be standard Destiny format as well
def overlay(img, name_str, type_str, flavor_str):
    std_w, std_h = (1920, 1080)
    img = img.resize((std_w, std_h))

    name_str = name_str.upper()
    type_str = type_str.upper()

    draw = ImageDraw.Draw(img)
    name_font = ImageFont.truetype("./fonts/title.ttf", 71)
    type_font = ImageFont.truetype("./fonts/title.ttf", 28)
    flvr_font = ImageFont.truetype("./fonts/flavor.otf", 28)

    draw.text((390, 120), name_str, (255, 255, 255), font = name_font)
    draw.text((390, 200), type_str, (202, 198, 197), font = type_font)
    draw.text((261, 263), flavor_str, (202, 198, 197), font = flvr_font)
    
    return img

def make_gun(g):
    n_models = g.num_models

    skip_mapping = False

    # Random latent vector and model
    latent = torch.randn(LATENT_DIM).cuda()
    model_index = random.randint(0, n_models - 1) * 1.0
    g.update_state_dict(model_index)

    # Generate and cast to something pygame can show
    sample = g.generate(latent, skip_mapping)
    trasnf = ToPILImage()
    img = trasnf(sample).convert("RGB")

    img = overlay(img, *get_item_txt())
    return img

WEAPON_TYPE = "HC"
g = Gen2("./models/" + WEAPON_TYPE + "/", "gparams.pt")

for name in range(1, 2):
    make_gun(g).save("test" + str(name) + ".png")
