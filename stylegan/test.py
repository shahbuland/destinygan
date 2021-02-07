from models import Generator, Discriminator

# In paper:
# Mapping network has 8 layers
# Generator has 2 layers for each resolution (1024^2 to 4^2)
# Latent space has dimentionality of 512
g = Generator(512, 8)

d = Discriminator()

