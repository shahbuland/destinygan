from train import train
from models import Generator, Discriminator, SwagDiscriminator
from constants import *

path = "datasets/pokemon/"
g = Generator()
d = SwagDiscriminator() if SWAGAN else Discriminator()
g.cuda()
d.cuda()
train(g, d, path)
