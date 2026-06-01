# Finite differences methods

import numpy as np


def shift2d(f, s, axis=0):
    g = np.roll(f, -s, axis)
    if s == 0:
        return f
    if axis == 0:
        if s > 0:
            for i in range(1, int(np.abs(s)) + 1):
                g[-i:, :] = f[-1, :]
        else:
            for i in range(0, int(np.abs(s))):
                g[i, :] = f[0, :]
    elif axis == 1:
        if s > 0:
            for i in range(1, int(np.abs(s)) + 1):
                g[:, -i] = f[:, -1]
        else:
            for i in range(0, int(np.abs(s))):
                g[:, i] = f[:, 0]

    return g


def shift(f, s, axis=0, **kwargs):
    n = np.abs(s, dtype=np.int32)
    fpad = np.pad(f, n, **kwargs)
    
    return np.roll(fpad, -s, axis)[*([slice(n, -n)]*len(f.shape))]
    


def d(f, h, axis=0, **kwargs):
    f_p1 = shift(f, 1, axis, **kwargs)
    f_0 = f
    f_m1 = shift(f, -1, axis, **kwargs)

    return 1/h**2 * (f_p1 - 2 * f_0 + f_m1)



def lap2D(f, hx, hz, **kwargs):
    """
    Second-order 2D laplacian operator
    """
    ix_p1 = 0
    ix_0 = 0
    ix_m1 = 0

    return d(f, hx, 0, **kwargs) + d(f, hz, 1, **kwargs)