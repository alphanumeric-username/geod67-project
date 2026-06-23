import numpy as np

def ricker(fpeak, dt, nt):
    t = np.arange(nt) * dt
    t0 = 1/fpeak
    t_ = t - t0
    return np.float32((1 - 2 * np.pi**2 * fpeak**2 * t_**2) * np.exp(-np.pi**2 * fpeak**2 * t_**2))
