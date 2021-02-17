import torch
from torch import nn
from torchvision.datasets import ImageFolder

from augment import AugmentLayer
import util
import loss_funcs
from constants import *

def train(g, d, data_path):
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    # Load dataset
    print("Loading dataset...")
    train_data = torch.utils.data.DataLoader(ImageFolder(data_path, util.imageToTensorTransform),
            batch_size = BATCH_SIZE, shuffle = True, num_workers = 4, pin_memory = True)

    scaler = torch.cuda.amp.GradScaler() # Mixed precision

    g_opt = torch.optim.AdamW(g.parameters(),
            lr = G_PPL_INTERVAL / (G_PPL_INTERVAL + 1),
            betas = (0, 0.99 ** (G_PPL_INTERVAL / (G_PPL_INTERVAL + 1))))
    d_opt = torch.optim.AdamW(d.parameters(),
            lr = D_R1REG_INTERVAL / (D_R1REG_INTERVAL + 1),
            betas = (0, 0.99 ** (D_R1REG_INTERVAL / (D_R1REG_INTERVAL + 1))))

    g.train()
    d.train()
    aug = AugmentLayer()

    w_bar = 0 # Avg PPL
    w_past = None # past path lengths

    for ITER in range(ITERATIONS + 1): 
        # Things on intervals
        do_d1_reg = ITER % D_R1REG_INTERVAL
        do_g_ppl = ITER % G_PPL_INTERVAL
        do_checkpoint = ITER % CHECKPOINT_INTERVAL
        do_sample = ITER % SAMPLE_INTERVAL
        do_log = ITER % LOG_INTERVAL

        # Discrminator forward
        real = next(train_data)
        latent = util.getMixingLatent(BATCH_SIZE)

        util.freezeModel(g)
        util.unfreezeModel(d)

        fake, _ = g(latent)
        
        real_orig = None
        if DO_AUGMENT:
            real_orig = real.copy()
            real = aug(real_orig)
            fake = aug(fake)

        fake_labels = d(fake)
        real_labels = d(real)

        # Discriminator backward
        d_loss = loss_funcs.discLoss(real_labels, fake_labels)
        d.zero_grad(set_to_none = True)
        d_loss.backward()
        d_opt.step()

        if DO_AUGMENT:
            aug.tune(real_labels)

        if do_d1_reg:
            real.requires_grad = True
            if DO_AUGMENT:
                real = aug(real_orig)
            
            real_labels = d(real)
            r1_loss = loss_funcs.r1Reg(real_orig, real_labels)

            d.zero_grad(set_to_none = True)
            r1_loss = R1REG_GAMMA / 2 * r1_loss * D_R1REG_INTERVAL
            r1_loss += 0 * real_labels[0] # Ties to d output
            r1_loss.backward()
            d_opt.step()

        util.unfreezeModel(g)
        util.freezeModel(g)

        latent = util.getMixingLatent(BATCH_SIZE)
        fake, _ = g(latent)

        if DO_AUGMENT:
            fake = aug(fake)

        fake_labels = d(fake)
        
        g_loss = loss_funcs.genLoss(fake_labels)
        g.zero_grad(set_to_none = True)
        g_loss.backward()
        g_opt.step()

        if do_g_ppl:
            path_batch = BATCH_SIZE // PPL_BATCH
            path_batch = max(0, path_batch)
            latent = util.getMixingLatent(path_batch)
            
            fake, latents = g(latent)
            ppl_loss, ppl_norms, w_bar = loss_funcs.PPL(latents, fake, w_bar)
            ppl_loss = PPL_WEIGHT * ppl_loss * G_PPL_INTERVAL
            ppl_loss += 0 * fake[0, 0, 0, 0] # Ties to g output

            g.zero_grad(set_to_none = True)
            ppl_loss.backward()
            g_opt.step()

            





