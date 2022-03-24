import cv2
import torch
import util
import augment
from PIL import Image

im = Image.open("gun.jpg")

transform = util.getImToTensorTF()

t = transform(im).cuda()

batch = torch.cat([t.unsqueeze(0) for _ in range(32)])

aug = augment.AugmentLayer()

batch = aug(batch)

util.drawSamples(batch, "./aug_res.png")
