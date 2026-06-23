import numpy as np


def generate_damp_mask(x, nbl):
    mask = np.ones_like(x[0, nbl:-nbl, nbl:-nbl, 0])
    mask = np.pad(mask, nbl, mode='linear_ramp')
    mask = 1 - mask
    mask = mask**4
    return mask.reshape((1, mask.shape[0], mask.shape[1], 1))