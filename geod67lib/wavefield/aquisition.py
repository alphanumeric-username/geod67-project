import tensorflow as tf

import numpy as np

class AquisitionParameters:
    pass

class AquisitionParameters:
    def __init__(self, src_positions, rec_positions, dt, tn, src_wavelet):
        self.src_positions = src_positions
        self.rec_positions = rec_positions
        self.dt = dt
        self.tn = tn//dt * dt

        self.src_wavelet = src_wavelet
        self.rec_data = np.zeros((self.rec_positions.shape[0], self.nt), dtype=src_wavelet.dtype)


    @property
    def nt(self):
        return int(self.tn//self.dt)
    

    def src(self, ti):
        if len(self.src_positions.shape) == 2:
            if ti < 0 or ti >= len(self.src_wavelet[0]):
                return 0
            return self.src_wavelet[:, ti]
        elif ti < 0 or ti >= len(self.src_wavelet):
            return 0
        return self.src_wavelet[ti]
    

    def update_rec_data(self, ti, u):
        for r in range(self.rec_positions.shape[0]):
            # self.rec_data[r, ti] = u[self.rec_positions[r]]
            xr = tuple(self.rec_positions[r].tolist())
            self.rec_data[r, ti] = u[xr]

    

    def adjoint_parameters(self):
        """
        Swaps
        """
        return AquisitionParameters(self.rec_positions, self.src_positions, self.dt, self.tn, self.rec_data)