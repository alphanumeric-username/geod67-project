import tensorflow as tf
from geod67lib.wavefield.aquisition import AquisitionParameters
from geod67lib.wavefield.model import VelocityModel
from geod67lib import fd

import numpy as np

class ForwardOperator:
    def __init__(self, aquisition_parameters: AquisitionParameters, damp_mask: np.ndarray):
        self.aquisition_parameters = aquisition_parameters
        self.damp_mask = damp_mask
        # self.u_next = tf.zeros_like(vp)
        # self.u = tf.zeros_like(vp)
        # self.u_prev = tf.zeros_like(vp)

    def forward(self, vp: VelocityModel, save_wavefield = False):
        u_next = np.zeros_like(vp.data)
        u = np.zeros_like(vp.data)
        u_prev = np.zeros_like(vp.data)

        if save_wavefield:
            u_history = []

        nt = self.aquisition_parameters.nt
        dt = self.aquisition_parameters.dt
        # src = lambda ti: self.aquisition_parameters.src(ti)

        for ti in range(nt):
            # print(ti, '/', nt)
            u_next = vp.data**2 * dt**2 * fd.lap2D(u, vp.hx, vp.hz) + 2 * u - u_prev#  - dt * (u - u_prev) * self.damp_mask
            # u_next = vp.data**2 * dt**2 * fd.lap2D(u, vp.hx, vp.hz) + 2 * u - u_prev  - dt * (u - u_prev) * self.damp_mask
            # u_next = vp.data**2 * dt**2 * fd.lap2D(u, vp.hx, vp.hz, mode='constant') + 2 * u - u_prev  - dt * (u - u_prev) * self.damp_mask
            u_next += -dt * self.damp_mask * (u_next - u_prev)/2
            src_i = self.aquisition_parameters.src(ti)
            for j in range(self.aquisition_parameters.src_positions.shape[0]):
                xs = tuple(self.aquisition_parameters.src_positions[j].tolist())
                u_next[xs] += vp.data[xs] **2 * dt**2 * src_i[j]

            if np.sum(np.isnan(u_next)) > 0:
                print('Nan')
                return u_history

            u_prev = u
            u = u_next

            self.aquisition_parameters.update_rec_data(ti, u)

            if save_wavefield:
                u_history.append(u)
        
        if save_wavefield:
            return u_history
        return []


    def adjoint(self):
        return ForwardOperator(self.aquisition_parameters.adjoint_parameters())