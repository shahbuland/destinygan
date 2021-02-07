import cv2
import torch
import util
import augment

im = cv2.imread("techtip.jpg")
print(im.shape)

t = util.imageToTensorCV(im)

augs = [augment.intTranslate for _ in range(10)]

im = util.tensorToCV(t)

im_augs = [util.tensorToCV(aug(t)) for aug in augs]

cv2.imshow('tech tip', im)
cv2.waitKey(0)
for im_aug in im_augs:
    cv2.imshow('aug', im_aug)
    cv2.waitKey(0)
cv2.destroyAllWindows()
