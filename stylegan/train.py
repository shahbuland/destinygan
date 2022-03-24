import torch
from torch import nn
from torchvision.datasets import ImageFolder
import torchvision

from augment import AugmentLayer
import util
import loss_funcs
from constants import *
from clock import Clock
from dataloader import MyDataLoader

import time
import copy

import wandb

def make_log(data):
    data = str(data)
    with open("./log", "w") as file:
              file.write(data)

def load_log():
    try:
        with open("./log", "r") as file:
                  data = file.read()
        return int(data)
    except:
        return None

# ==== UTILITIES FOR GRAD ACCUM TO BE SIMPLER ====

# f: batch -> (loss for training, loss for logging)
# or
# f: batch -> (loss for training,
#               loss for logging, 
#               extra term)
# returns standardized loss term + list of extra terms
def grad_accum(f, batch, opt):
    n, _, _, _ = batch.shape
    num_steps = n // MICROBATCH_SIZE
    mbs = torch.chunk(batch, num_steps)
    total_loss = 0
    extras = []
    opt.zero_grad(set_to_none = True)
    for mb in mbs:
        ret = f(batch)
        if len(ret) == 3:
            loss, loss_log, extra = ret
            extras.append(extra)
        else:
            loss, loss_log = ret
            
        total_loss += loss_log
        (loss / num_steps).backward()
    opt.step()
    return total_loss / BATCH_SIZE, extras
        
# Methods below expect batches to not be augmented
def get_d_loss(g, d, batch, aug = None, do_r1_reg = False):
    n, _, _, _ = batch.shape
    if do_r1_reg: batch.requires_grad = True
    if USE_AUGMENTS:
        batch = aug(batch)
    with torch.no_grad():
        fake, _ = g(util.getMixingLatent(n))
        if USE_AUGMENTS:
            fake = aug(fake)
     
    fake_labels = d(fake)
    real_labels = d(batch)

    d_loss = loss_funcs.discLoss(real_labels, fake_labels)
    d_adv_loss = d_loss.item()
    if do_r1_reg:
        d_loss += R1REG_GAMMA * loss_funcs.r1Reg(batch, real_labels)
    return d_loss, d_adv_loss 

# Batch should be "dummy" (this just streamlines grad accum)
def get_g_loss(g, d, batch, aug = None, do_path_reg = False, mean_pl = None):
    n, _, _, _ = batch.shape
    fake, latents = g(util.getMixingLatent(n))
    if USE_AUGMENTS:
        labels = d(aug(fake))
    else:
        labels = d(fake)
    g_loss = loss_funcs.genLoss(labels)
    g_adv_loss = g_loss.item()
    avg_pl = None
    if do_path_reg:
        path_lengths = loss_funcs.pathLength(latents, fake)
        avg_pl = path_lengths.mean()
        if mean_pl is not None:
            pl_loss = ((path_lengths - mean_pl) ** 2).mean()
            g_loss += pl_loss
    return g_loss, g_adv_loss, avg_pl

# ==============================================

def train(g, g_ema, d, data_path, train_offset = 0):
    if USE_WANDB:
        wandb.init(project='swagandiffaug', entity='shahbuland')
        wandb.watch(g)
        wandb.watch(d)
    
    if DO_EMA:
        # Exp moving avg of generator
        if g_ema is None:
            g_ema = copy.deepcopy(g)
        # it won't be trained
        g_ema.eval()
        util.freezeModel(g_ema)

    # Try to load log file
    train_offset = load_log()

    # Load dataset
    print("Loading dataset...")
    if LOAD_DATA == "SOME":
        data_path = "./datasets/" + data_path
        dataset = ImageFolder(data_path, transform = util.getImToTensorTF())
        train_data = torch.utils.data.DataLoader(dataset, batch_size = BATCH_SIZE,
                shuffle = True, pin_memory = True, num_workers = 0)
    else:
        print("Putting data on RAM (may take a minute)")
        train_data = MyDataLoader(data_path)

    g_opt = torch.optim.AdamW(g.parameters(),
            lr = LEARNING_RATE * G_PPL_INTERVAL / (G_PPL_INTERVAL + 1),
            betas = (0, 0.99 ** (G_PPL_INTERVAL / (G_PPL_INTERVAL + 1))))
    d_opt = torch.optim.AdamW(d.parameters(),
            lr = LEARNING_RATE * D_R1REG_INTERVAL / (D_R1REG_INTERVAL + 1),
            betas = (0, 0.99 ** (D_R1REG_INTERVAL / (D_R1REG_INTERVAL + 1))))

    g.train()
    d.train()
    aug = AugmentLayer()

    mean_pl = None # overal mean PPL

    timer = Clock()

    for ITER in range(train_offset, ITERATIONS + 1):
        # Things on intervals
        do_r1_reg = (ITER % D_R1REG_INTERVAL == 0) and DO_R1_REG
        do_g_ppl = (ITER % G_PPL_INTERVAL == 0) and DO_PPL_REG
        do_checkpoint = ITER % CHECKPOINT_INTERVAL == 0
        do_sample = ITER % SAMPLE_INTERVAL == 0
        do_log = ITER % LOG_INTERVAL == 0
        
        real = next(iter(train_data))[0] if LOAD_DATA == "SOME" else train_data.next()
        real = real.to(DEVICE)

        # Train discriminator
        util.freezeModel(g)
        util.unfreezeModel(d)
        
        d_loss_adv, _ = grad_accum(lambda batch: get_d_loss(g, d, batch, aug, do_r1_reg),
                                real, d_opt)
        d_opt.step()
        # Train generator
        util.freezeModel(d)
        util.unfreezeModel(g)
        g_loss_adv, new_pl_avg = grad_accum(lambda batch: get_g_loss(g, d, batch, aug, do_g_ppl, mean_pl),
                                real, g_opt)
        if do_g_ppl:
            new_pl_avg = torch.stack(new_pl_avg).mean()
            mean_pl = new_pl_avg if mean_pl is None else \
                        mean_pl * (1 - PPL_DECAY) + PPL_DECAY * new_pl_avg
         
        # exp moving average of g
        if DO_EMA: util.get_ema(g_ema, g)

        if do_log:
            time_per_k = timer.hit() // (LOG_INTERVAL * BATCH_SIZE)
            im_shown = (ITER * BATCH_SIZE) / 1e6
            im_shown = round(im_shown, 2)
            print("[" + str(ITER) + "/" + str(ITERATIONS) + "] D Loss: " 
                + str(d_loss_adv) + ", G Loss: " + str(g_loss_adv)
                    + ", Avg Time (Per 1k Shown): " + str(time_per_k) + "s" +
                  ", Images Shown: " + str(im_shown) + "M")
            if USE_WANDB:
                wandb.log({"G Loss":g_loss_adv,"D Loss":d_loss_adv})

            if DO_EMA:
                with torch.no_grad():
                    samples_ema = g_ema.generate(MICROBATCH_SIZE)
                util.drawSamples(samples_ema, "samples_ema.png")
            
            with torch.no_grad():
                samples = g.generate(MICROBATCH_SIZE)
            util.drawSamples(samples, "samples.png")
            
            torch.save(g.state_dict(), "./gparams.pt")
            if DO_EMA: torch.save(g_ema.state_dict(), "./gemaparams.pt")
            torch.save(d.state_dict(), "./dparams.pt")
            make_log(ITER+1)
        
        if do_sample:
            with torch.no_grad():
                if DO_EMA and False:
                    samples = g_ema.generate(MICROBATCH_SIZE)
                else:
                    samples = g.generate(MICROBATCH_SIZE)
            util.drawSamples(samples, "./samples/" + str(ITER) + ".png")

        if do_checkpoint:
            torch.save(g.state_dict(), "./checkpoints/" + str(ITER) + "gparams.pt")
            if DO_EMA: torch.save(g_ema.state_dict(), "./checkpoints/" + str(ITER) + "gemaparams.pt")
            torch.save(d.state_dict(), "./checkpoints/" + str(ITER) + "dparams.pt")
