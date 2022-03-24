import torch
from torch import nn
import torch.nn.functional as F

import util

# Loss for search is slightly different
# Must move away from non matching embeddings
# Move toward matching ones
# Assumes x is being searched, y is fixed
# y_index is index of what is being searched for
def searchLoss(x, y, y_index, logit_scale):
    n = x.shape[0]

    x = F.normalize(x)
    y = F.normalize(y)

    logits = x @ y.T * logit_scale
    labels = torch.ones(n) * y_index
    
    return F.cross_entropy(logits, labels), logits

c_batch = 128
micro_c_batch = 8

n_search = 32
learning_rate = 1e-3
n_steps = 128

c_latent = 512
g_latent = 512


# Search icon generator latent space for single gun
# using contraster embeddings
def latentSearch(generator, contraster, gun):
    # Freeze params
    for p in generator.parameters():
        p.requires_grad = False
    for p in contraster.parameters():
        p.requires_grad = False

    # Prepare gun encodings
    loader = util.DataLoader(c_batch - 1)
    gun_batch, _ = loader.next()
    gun_batch = torch.cat([gun.unsqueeze(0), gun_batch])
    gun_batch = torch.chunk(gun_batch, c_batch // micro_c_batch)
    with torch.no_grad():
        gun_encs = [contraster.encodeGun(gun_mb) for gun_mb in gun_batch]
        gun_encs = torch.cat(gun_encs)
    
    # Get latents and prepare them for optimization
    latents = torch.randn(n_search, g_latent)
    latents.requires_grad = True
    opt = torch.optim.AdamW([latents], lr = learning_rate)

    # Training loop
    for step in range(n_steps):
        ico_batch = generator.generate(latents)
        ico_encs = contraster.encodeIco(ico_batch)
        loss, _ = searchLoss(ico_encs, gun_encs, 0, contraster.logit_scale)

        loss.backward()
        opt.step()

    with torch.no_grad():
        ico_batch = generator.generate(latents)
        ico_encs = contraster.encodeIco(ico_batch)
        _, logits = searchLoss(ico_encs, gun_encs, 0, contraster.logit_scale)
        scores = logits[:,0]
        return latents[torch.argmax(scores)]

    


