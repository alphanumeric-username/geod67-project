import tensorflow as tf
from geod67lib.wavefield.aquisition import AquisitionParameters
from geod67lib.wavefield.model import VelocityModel
from geod67lib import fd
from geod67lib.timer import Timer

import numpy as np

def dirac(pos, shape):
    d = np.zeros(shape, dtype=np.float32)
    d[pos] = 1
    return tf.constant(d)


def multiple_dirac(pos,shape,values):
    d = np.zeros(shape, dtype=np.float32)
    for i, p in enumerate(pos):
        di = np.zeros(shape, dtype=np.float32)
        di[p] = 1
        d += di * values[i]
        # print(values[i], d.shape)
    return d



class WaveSolverOperator:
    def __init__(self, aquisition_parameters: AquisitionParameters):
        self.aquisition_parameters = aquisition_parameters


    def forward(self, vp: VelocityModel, damp_mask: tf.Tensor = None, save_wavefield = False, fd_order=2):
        u_next = tf.zeros_like(vp.data)
        u = tf.zeros_like(vp.data)
        u_prev = tf.zeros_like(vp.data)

        dop = fd.DifferentialOperator(fd_order, vp.hz, vp.hz)

        if save_wavefield:
            u_history = []

        nt = self.aquisition_parameters.nt
        dt = self.aquisition_parameters.dt

        pre_values = []
        src_values = []
        pos = []
        for j in range(self.aquisition_parameters.src_positions.shape[0]):
            xs = np.array(self.aquisition_parameters.src_positions[j], dtype=np.int32).tolist()
            xs.insert(0, 0)
            xs.append(0)
            xs = tuple(xs)
            pos.append(xs)
            pre_values.append(vp.data[xs] * vp.data[xs] * dt*dt)

        for ti in range(nt):
            v = []
            for j in range(self.aquisition_parameters.src_positions.shape[0]):
                src_ti = self.aquisition_parameters.src(ti)
                v.append(pre_values[j] * src_ti[j])
            src_values.append(tf.SparseTensor(pos, v, u_next.shape))

        dt2 = dt * dt
        vp_data2 = vp.data * vp.data
        vp_data2_dt2 = vp_data2 * dt2
        if not(damp_mask is None):
            dt_damp_mask_5 = -dt * damp_mask * .5

        for ti in range(nt):
            u_prev, u, u_next = self._update_wavefield(u_prev, u, u_next, src_values[ti], vp_data2_dt2, dop, not(damp_mask is None), dt_damp_mask_5)
            self.aquisition_parameters.update_rec_data(ti, u)

            if save_wavefield:
                u_history.append(u)
        
        if save_wavefield:
            return u_history
        return []


    @tf.function
    def _update_wavefield(self, u_prev, u, u_next, src_ti, vp_data2_dt2, dop, use_damp_mask, dt_damp_mask_5):
        u_next = vp_data2_dt2 * dop.lap(u) + 2 * u - u_prev
        if use_damp_mask:
            u_next += dt_damp_mask_5 * (u_next - u_prev)
            
        u_next = tf.sparse.add(u_next, src_ti)
        u_prev = u
        u = u_next

        return u_prev, u, u_next



    def __call__(self, *args, **kwargs):
        self.forward(*args, **kwargs)
    

    def adjoint(self):
        return WaveSolverOperator(self.aquisition_parameters.adjoint_parameters())