import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from constants import *
import util

from torch_utils.ops import upfirdn2d, grid_sample_gradfix, conv2d_gradfix

sym6wavelet = [0.015404109327027373, 0.0034907120842174702, -0.11799011114819057,
        -0.048311742585633, 0.4910559419267466, 0.787641141030194,
        0.3379294217276218, -0.07263752278646252, -0.021060292512300564,
        0.04472490177066578, 0.0017677118642428036, -0.007800708325034148]
DEVICE = 'cuda'

# Augment layer that has a chance to apply augmentation
class AugmentLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fir_k = upfirdn2d.setup_filter(sym6wavelet).cuda()
        # For adaptive part
        self.aug_prob = torch.tensor([AUG_PROB], device = DEVICE)
        self.updates = 0
        self.r_t = 0
        # Sum of all discriminator outputs and overall number of outputs
        # Used to get expectation (mean)
        self.d_outs = torch.Tensor([0])
        self.num_outs = 0

    def adapt(self, real_labels):
        self.d_outs += torch.sign(real_labels).sum().item()
        self.num_outs += real_labels.shape[0]
        self.updates += 1
        return
        #return
        if self.updates % ADA_INTERVAL == 0:
            self.r_t = self.d_outs / self.num_outs

            if self.r_t > AUG_P_TARGET:
                sign = 1
            else:
                sign = -1

            one = torch.tensor([1], device = DEVICE)
            zer = torch.tensor([0], device = DEVICE)
            
            self.aug_prob += sign * self.num_outs / ADA_LENGTH
            self.aug_prob = min(self.aug_prob, one)
            self.aug_prob = max(self.aug_prob, zer)

            self.d_outs *= 0
            self.num_outs = 0

    def forward(self, x):
        I_3 = torch.eye(3, device = DEVICE)
        G_inv = I_3 # inverse of G in paper

        batch_size, ch, h, w = x.shape
        
        # x-flip
        if AUG_XFLIP > 0:
            i = prob_mask(torch.rand, [batch_size], 2,
                    torch.floor, self.aug_prob * AUG_XFLIP)
            G_inv = G_inv @ scale2D_inv(1 - 2 * i, 1)

        # 90 deg rotations
        if AUG_ROT90 > 0:
            i = prob_mask(torch.rand, [batch_size], 4,
                    torch.floor, self.aug_prob * AUG_ROT90)
            G_inv = G_inv @ rotate2D_inv(-np.pi * i / 2)

        # int translation
        if AUG_XINT > 0:
            t = prob_mask(torch.rand, [batch_size], 2,
                    lambda x : x - 1, self.aug_prob * AUG_XINT,
                    num_vars = 2)
            t *= 0.125
            t_x = torch.round(t[:, 0] * w)
            t_y = torch.round(t[:, 1] * h)
            G_inv = G_inv @ translate2D_inv(t_x, t_y)

        # isotropic scale
        if AUG_SCALE > 0:
            s = prob_mask(torch.randn, [batch_size], 0.2,
                    torch.exp2, self.aug_prob * AUG_SCALE,
                    fill = torch.ones_like)
            G_inv = G_inv @ scale2D_inv(s, s)

        p_rot = 1 - torch.sqrt((1 - self.aug_prob * AUG_ROT).clamp(0, 1))
        # Rotation before AT scaling
        if AUG_ROT > 0:
            theta = prob_mask(torch.rand, [batch_size], 2,
                    lambda x : x - 1, p_rot)
            theta *= np.pi
            G_inv = G_inv @ rotate2D_inv(-theta)

        # Anisotropic scale
        if AUG_ANISO > 0:
            s = prob_mask(torch.randn, [batch_size], 0.2,
                    torch.exp2, self.aug_prob * AUG_ANISO,
                    fill = torch.ones_like)
            G_inv = G_inv @ scale2D_inv(s, 1 / s)

        # Rotation after AT scale
        if AUG_ROT > 0:
            theta = prob_mask(torch.rand, [batch_size], 2,
                    lambda x : x - 1, p_rot)
            theta *= np.pi
            G_inv = G_inv @ rotate2D_inv(-theta)

        # Fractional translation
        if AUG_XFRAC > 0:
            t = prob_mask(torch.randn, [batch_size], 0.125,
                    lambda x : x, self.aug_prob * AUG_XFRAC,
                    num_vars = 2)
            t_x = t[:, 0]
            t_y = t[:, 1]
            G_inv = G_inv @ translate2D_inv(t_x * w, t_y * h)

        # No need to execute if G is still I_3 (i.e. no change)
        if G_inv is not I_3:
            # Orthogonal low pass filter
            H_pad = self.fir_k.shape[0] // 4
             
            # Padding
            m_x0, m_y0, m_x1, m_y1 = calculatePadding(G_inv, w, h, H_pad)
            pad = [m_x0, m_x1, m_y0, m_y1]
            x = F.pad(x, pad, mode = 'reflect')

            t_x = (m_x0 - m_x1) / 2
            t_y = (m_y0 - m_y1) / 2
            t_x = torch.ones(batch_size, device = DEVICE) * t_x
            t_y = torch.ones(batch_size, device = DEVICE) * t_y
            T = translate2D(t_x, t_y)

            G_inv = T @ G_inv
            
            # Upsample
            x = upfirdn2d.upsample2d(x, f = self.fir_k, up = 2)
            
            s_param = torch.ones(batch_size, device = DEVICE) * 2
            S_in = scale2D(s_param, s_param)
            S_out = scale2D_inv(s_param, s_param)
            
            G_inv = S_in @ G_inv @ S_out
           
            t_xy = torch.ones(2, batch_size) * -.5
            T_in = translate2D(t_xy[0], t_xy[1])
            T_out = translate2D_inv(t_xy[0], t_xy[1])
            
            G_inv = T_in @ G_inv @ T_out

            # Execute
            shape = [batch_size, ch, (h + H_pad * 2) * 2, (w + H_pad * 2) * 2]
           
            s_x = torch.ones(batch_size, device = DEVICE) * \
                    2 / x.shape[3]
            s_y = torch.ones(batch_size, device = DEVICE) * \
                    2 / x.shape[2]
            s_x_out = torch.ones(batch_size, device = DEVICE) * \
                    2 / shape[3]
            s_y_out = torch.ones(batch_size, device = DEVICE) * \
                    2 / shape[2]

            S_in = scale2D(s_x, s_y)
            S_out = scale2D_inv(s_x_out, s_y_out)
            
            G_inv = S_in @ G_inv @ S_out
            
            grid = F.affine_grid(theta = G_inv[:,:2,:], size = shape, align_corners = False)
            x = grid_sample_gradfix.grid_sample(x, grid)

            # Down and crop
            x = upfirdn2d.downsample2d(x = x, f = self.fir_k,
                    down = 2, padding = -2 * H_pad, flip_filter = True)

        # Color transforms
        I_4 = torch.eye(4, device = DEVICE)
        C = I_4

        # Brightness
        if AUG_BRIGHTNESS > 0:
            b = prob_mask(torch.randn, [batch_size], 0.2,
                    lambda x : x, self.aug_prob * AUG_BRIGHTNESS)
            C = translate3D(b, b, b) @ C

        # Contrast
        if AUG_CONTRAST > 0:
            c = prob_mask(torch.randn, [batch_size], 0.5,
                    torch.exp2, self.aug_prob * AUG_CONTRAST,
                    fill = torch.ones_like)
            C = scale3D(c, c, c) @ C

        # Luma axis
        v = torch.tensor([1, 1, 1, 0], device = DEVICE) / (3**.5)

        # Luma flip
        if AUG_LUMAFLIP > 0:
            i = prob_mask(torch.rand, [batch_size], 2,
                    torch.floor, self.aug_prob * AUG_LUMAFLIP)
            i = i.unsqueeze(1).unsqueeze(2) # B x 1 x 1
            C = lumaFlip3D(v, i) @ C

        # Hue rotation
        if AUG_HUE > 0:
            theta = prob_mask(torch.rand, [batch_size], 2,
                    lambda x : x - 1, self.aug_prob * AUG_HUE)
            theta *= np.pi
            C = rotate3D(v, theta) @ C

        # Saturation
        if AUG_SAT > 0:
            s = prob_mask(torch.randn, [batch_size], 1,
                    torch.exp2, self.aug_prob * AUG_SAT,
                    fill = torch.ones_like)
            s.unsqueeze(1).unsqueeze(2) # B x 1 x 1
            C = saturation3D(v, s) @ C

        # Execute if need to
        if C is not I_4:
            x = x.reshape([batch_size, ch, h * w])
            x = C[:, :3, :3] @ x + C[:, :3, 3:]
            x = x.reshape([batch_size, ch, h, w])

        # In paper, they say image space filtering didn't do anything
        # So I don't implement

        return x

    

wavelets = {
    'sym6' : [0.015404109327027373, 0.0034907120842174702, -0.11799011114819057, -0.048311742585633,
        0.4910559419267466, 0.787641141030194, 0.3379294217276218, -0.07263752278646252,
        -0.021060292512300564, 0.04472490177066578, 0.0017677118642428036, -0.007800708325034148]
}

def calculatePadding(G_inv, w, h, H_pad):
    cx = (w - 1) / 2
    cy = (h - 1) / 2
    cp = torch.tensor([
        [-cx, -cy, 1],
        [cx, -cy, 1],
        [cx, cy, 1],
        [-cx, cy, 1]], device = DEVICE)
    cp = G_inv @ cp.t()
    
    margin = cp[:, :2, :].permute(1, 0, 2).flatten(1) 
    margin = torch.cat([-margin, margin]).max(dim = 1).values
    margin = margin + torch.tensor([H_pad * 2 - cx, H_pad * 2 - cy] * 2,
            device = DEVICE)
    margin = margin.max(torch.tensor([0, 0] * 2, device = DEVICE))
    margin = margin.min(torch.tensor([w - 1, h - 1] * 2, device = DEVICE))

    return margin.ceil().to(torch.int32)

# Augmentation matrices

# get batch of identities
def batchID(n, batch):
    ret = torch.eye(n, device = DEVICE)
    ret = ret.unsqueeze(0).repeat(batch, 1, 1)
    return ret

# Geometric
def scale2D(s_x, s_y):
    ret = batchID(3, s_x.shape[0])
    ret[:, 0, 0] = s_x
    ret[:, 1, 1] = s_y

    return ret

def scale2D_inv(s_x, s_y):
    return scale2D(1 / s_x, 1 / s_y)

def translate2D(t_x, t_y):
    ret = batchID(3, t_x.shape[0])
    ret[:, 0, 2] = t_x
    ret[:, 1, 2] = t_y

    return ret

def translate2D_inv(t_x, t_y):
    return translate2D(-t_x, -t_y)

def rotate2D(theta):
    ret = batchID(3, theta.shape[0])
    s = torch.sin(theta)
    c = torch.cos(theta)
    ret[:, 0, 0] = c
    ret[:, 0, 1] = -s
    ret[:, 1, 0] = s
    ret[:, 1, 1] = c
    return ret

def rotate2D_inv(theta):
    return rotate2D(-theta)

# Color

def scale3D(s_x, s_y, s_z):
    ret = batchID(4, s_x.shape[0])
    ret[:, 0, 0] = s_x
    ret[:, 1, 1] = s_y
    ret[:, 2, 2] = s_z
    return ret

def translate3D(t_x, t_y, t_z):
    ret = batchID(4, t_x.shape[0])
    ret[:, 0, 3] = t_x
    ret[:, 1, 3] = t_y
    ret[:, 2, 3] = t_z
    return ret

# 3D theta rad rotation matrix about axis v
def rotate3D(v, theta):
    vx, vy, vz, _ = v
    s = torch.sin(theta)
    c = torch.cos(theta)
    cc = 1 - c

    ret = batchID(4, theta.shape[0])

    ret[:, 0, 0] = vx * vx * cc + c
    ret[:, 0, 1] = vx * vy * cc - vz * s 
    ret[:, 0, 2] = vx * vz * cc + vy * s
    ret[:, 1, 0] = vy * vx * cc + vy * s
    ret[:, 1, 1] = vy * vy * cc + c
    ret[:, 1, 2] = vy * vz * cc - vx * s
    ret[:, 2, 0] = vz * vx * cc - vy * s
    ret[:, 2, 1] = vz * vy * cc + vx * s
    ret[:, 2, 2] = vz * vz * cc + c

    return ret

def lumaFlip3D(v, i):
    i_4 = batchID(4, i.shape[0])

    return (i_4 - 2 * v.outer(v) * i)

def saturation3D(v, i):
    i_4 = batchID(4, i.shape[0])
    v = v.outer(v)

    return v + (i_4 - v) * i.view(-1, 1, 1)

# Samples fn(mult * X) from X ~ dist for [size]
# Then (random mask) sets individual elements to 0 with prob (1 - p),
# where p is
# p = P(Y < thresh), Y ~ U(0, 1)
# (size should be list)
# Used to parameterize transformation over a batch with random application chance
# Fill(i) should be parameters if transformation isn't used
def prob_mask(dist, size, mult, fn, thresh, fill = torch.zeros_like, num_vars = 1):
    # Get parameters for transformation
    if num_vars > 1:
        i = dist(size + [num_vars], device = DEVICE)
    else:
        i = dist(size, device = DEVICE)
    i = mult * i
    i = fn(i)

    if num_vars > 1:
        size += [1]
    # Randomly choose which items in batch we apply transform to
    i = torch.where(torch.rand(size, device = DEVICE) < thresh, i, 
            fill(i))
    return i

