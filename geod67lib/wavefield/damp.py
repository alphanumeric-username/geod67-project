import numpy as np


def generate_damp_mask(x, nbl):
    mask = np.ones_like(x[nbl:-nbl, nbl:-nbl])
    mask = np.pad(mask, nbl, mode='linear_ramp')
    mask = 1 - mask
    mask = mask**4
    return mask