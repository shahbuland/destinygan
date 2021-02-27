import torch
from torch import nn
from torchvision.datasets import ImageFolder
import torchvision

from augment import AugmentLayer
import util
import loss_funcs
from constants import *

def train(g, d, data_path):
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    # Load dataset
    print("Loading dataset...")
    dataset = ImageFolder(data_path, util.imageToTensorTransform)
    train_data = torch.utils.data.DataLoader(dataset, batch_size = BATCH_SIZE,
            shuffle = True)

    #scaler = torch.cuda.amp.GradScaler() # Mixed precision

    g_opt = torch.optim.Adam(g.parameters(),
            lr = LEARNING_RATE * G_PPL_INTERVAL / (G_PPL_INTERVAL + 1),
            betas = (0, 0.99 ** (G_PPL_INTERVAL / (G_PPL_INTERVAL + 1))))
    d_opt = torch.optim.Adam(d.parameters(),
            lr = LEARNING_RATE * D_R1REG_INTERVAL / (D_R1REG_INTERVAL + 1),
            betas = (0, 0.99 ** (D_R1REG_INTERVAL / (D_R1REG_INTERVAL + 1))))

    g.train()
    d.train()
    aug = AugmentLayer()

    w_bar = 0 # Avg PPL
    w_past = None # past path lengths

    for ITER in range(ITERATIONS + 1): 
        # Things on intervals
        do_d1_reg = ITER % D_R1REG_INTERVAL == 0
        do_g_ppl = ITER % G_PPL_INTERVAL == 0
        do_checkpoint = ITER % CHECKPOINT_INTERVAL == 0
        do_sample = ITER % SAMPLE_INTERVAL == 0
        do_log = ITER % LOG_INTERVAL == 0
        
        # Discrminator forward
        real, _ = next(iter(train_data))
        real = real.cuda()
        latent = util.getMixingLatent(BATCH_SIZE)

        util.freezeModel(g)
        util.unfreezeModel(d)

        fake, _ = g(latent)
        
        real_orig = None
        if USE_AUGMENTS:
            real_orig = real.clone()
            real = aug(real_orig)
            fake = aug(fake)

        fake_labels = d(fake)
        real_labels = d(real)

        # Discriminator backward
        d_loss = loss_funcs.discLoss(real_labels, fake_labels)
        d_loss_adv = d_loss.item()
        d.zero_grad(set_to_none = True)
        d_loss.backward()
        d_opt.step()

        if USE_AUGMENTS:
            aug.adapt(real_labels)

        if do_d1_reg:
            real.requires_grad = True
            if USE_AUGMENTS:
                real = aug(real_orig)
                real_orig.requires_grad = True

            real_labels = d(real)
            r1_loss = loss_funcs.r1Reg(real_orig if USE_AUGMENTS else real, real_labels)

            d.zero_grad(set_to_none = True)
            r1_loss = R1REG_GAMMA / 2 * r1_loss * D_R1REG_INTERVAL
            r1_loss = r1_loss.item() + 0 * real_labels[0] # Ties to d output
            r1_loss.backward()
            d_opt.step()

        util.unfreezeModel(g)
        util.freezeModel(d)

        latent = util.getMixingLatent(BATCH_SIZE)
        fake, _ = g(latent)
        if USE_AUGMENTS:
            fake = aug(fake)

        fake_labels = d(fake)
        
        g_loss = loss_funcs.genLoss(fake_labels)
        g_loss_adv = g_loss.item()
        g_opt.zero_grad(set_to_none = True)
        g_loss.backward()
        g_opt.step()
        
        if do_g_ppl and False:
            path_batch = BATCH_SIZE // PPL_BATCH
            path_batch = max(1, path_batch)
            latent = util.getMixingLatent(path_batch)
            
            fake, latents = g(latent)
            ppl_loss, ppl_norms, w_bar = loss_funcs.PPL(latents, fake, w_bar)
            ppl_loss = PPL_WEIGHT * ppl_loss * G_PPL_INTERVAL
            ppl_loss += 0 * fake[0, 0, 0, 0] # Ties to g output

            print("1.1")
            g.zero_grad(set_to_none = True)
            print("1.2")
            ppl_loss.backward()
            print("1.3")
            g_opt.step()
            print("1.4")
            w_bar = w_bar.sum().item()
        
        if do_log:
            print("[" + str(ITER) + "/" + str(ITERATIONS) + "] D Loss: " + str(d_loss_adv) + ", G Loss: " + str(g_loss_adv))

        if do_sample:
            samples = g.generate(BATCH_SIZE)
            util.drawSamples(samples, "./samples/" + str(ITER) + ".png")
            #util.drawSamples(real, "./samples/" + str(ITER) + "aug.png")
            if USE_AUGMENTS: util.drawSamples(real_orig, "./samples/" + str(ITER) + "orig.png")

        if do_checkpoint:
            torch.save(g.state_dict(), "./checkpoints/" + str(ITER) + "gparams.pt")
            torch.save(d.state_dict(), "./checkpoints/" + str(ITER) + "dparams.pt")
