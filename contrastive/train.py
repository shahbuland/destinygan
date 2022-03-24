import torch
from augment import AugmentLayer
import util
from constants import *

def generate_indices(total_size, batch_size, shuffle = True):
    inds = torch.randperm(total_size) if shuffle else \
        torch.arange(total_size)
    return torch.chunk(inds, total_size // batch_size)

def get_scheduling_func():
    def lerp(a, b, t):
        t = min(1, t)
        t = max(0, t)
        return a + (b - 1) * t

    ratio = LEARNING_RATE_TARGET - LEARNING_RATE_INIT
    return lambda step: \
        (step + 1) / LR_RAMP_STEPS if step < LR_RAMP_STEPS \
        else lerp(1, ratio, (step - LR_RAMP_STEPS) / LR_DECAY__STEPS)

# Expects dataset to be (gun, ico)
# But paired (i.e. gun[i] matches ico[i]
def train(model):
    model.train()
    aug = AugmentLayer()
    loader = util.DataLoader(BATCH_SIZE)

    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    if LOAD_CHECKPOINT:
        model.load_state_dict(torch.load("params.pt"))

    for ITER in range(ITERATIONS + 1):
        gun_batch, ico_batch = loader.next()
        gun_batch = gun_batch.to('cuda')
        ico_batch = ico_batch.to('cuda')
        batch_inds = torch.arange(BATCH_SIZE)
        batch_inds = torch.chunk(batch_inds, BATCH_SIZE // MICROBATCH_SIZE)

        gun_mbs = [aug(gun_batch[mb_inds]) for mb_inds in batch_inds]
        ico_mbs = [aug(ico_batch[mb_inds]) for mb_inds in batch_inds]
        
        with torch.no_grad():
            gun_encs = [model.encodeGun(gun_mb) for gun_mb in gun_mbs]
            ico_encs = [model.encodeIco(ico_mb) for ico_mb in ico_mbs]
            
            train_loss, train_acc = model.cLoss(torch.cat(gun_encs),
                                                torch.cat(ico_encs))

        opt.zero_grad()
        for index, gun_mb in enumerate(gun_mbs):
            enc_tmp = gun_encs.copy()
            enc_tmp[index] = model.encodeGun(gun_mb)
            loss, _ = model.cLoss(torch.cat(enc_tmp),
                                    torch.cat(ico_encs))
            loss.backward()

        for index, ico_mb in enumerate(ico_mbs):
            enc_tmp = ico_encs.copy()
            enc_tmp[index] = model.encodeIco(ico_mb)
            loss, _ = model.cLoss(torch.cat(gun_encs),
                                    torch.cat(enc_tmp))
            loss.backward()

        opt.step()

        if ITER % LOG_INTERVAL == 0:
            print(str(ITER) + " Loss: " + str(train_loss.item()) + \
                    ", Acc: " + str(train_acc.item()))
        if ITER % CHECKPOINT_INTERVAL == 0:
            print("SAVING...")
            torch.save(model.state_dict(), "./params.pt")

from model import ContrastiveModel
from util import get_dataset, get_toy

if __name__ == "__main__":
    model = ContrastiveModel()
    model.cuda()

    train(model)
