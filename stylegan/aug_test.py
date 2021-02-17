import cv2
import torch
import util
import augment

im = cv2.imread("gun.jpg")
print(im.shape)

t = util.imageToTensorCV(im)

#augs = [augment.isotropicScale, augment.intTranslate, augment.randRotate, augment.xFlip]

im = util.tensorToCV(t)
augs = []

im_augs = [util.tensorToCV(aug(t)) for aug in augs]

cv2.waitKey(0)
for im_aug in im_augs:
    cv2.imshow('aug', im_aug)
    cv2.waitKey(0)

from torchvision.transforms import RandomAffine as RA
from torchvision.transforms import ColorJitter as CJ

rot = CJ(0, saturation = [0, 7])
aug_im = util.tensorToCV(rot(t))
cv2.imshow('aug', aug_im)
cv2.waitKey(0)
cv2.destroyAllWindows()
