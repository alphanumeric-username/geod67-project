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

        tm = Timer()
        tm2 = Timer()

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
            # pre_values.append(vp.data[xs] **2 * dt**2)
            pre_values.append(vp.data[xs] * vp.data[xs] * dt*dt)
        # pos = tf.constant(pos)

        for ti in range(nt):
            # values.append([])
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

        # tm2.start()
        for ti in range(nt):
            # if ((ti + 1) % 50) == 0:
            #     print(ti + 1, '/', nt)
            # u_next = vp.data**2 * dt**2 * fd.lap(u, vp.hx, vp.hz, fd_order=fd_order) + 2 * u - u_prev
            # u_next = vp.data**2 * dt**2 * dop.lap(u) + 2 * u - u_prev
            # print('Stencil:', end=' ')
            # tm.start()
            # u_next = vp.data*vp.data * dt*dt * dop.lap(u) + 2 * u - u_prev
            u_next = vp_data2_dt2 * dop.lap(u) + 2 * u - u_prev
            if not(damp_mask is None):
                # u_next += -dt * damp_mask * (u_next - u_prev)/2
                u_next += dt_damp_mask_5 * (u_next - u_prev)
            # print(tm.stop())
            
            # print('Source:', end='  ')
            # tm.start()
            # src_ti = self.aquisition_parameters.src(ti)
            # values = []
            # pos = []
            # for j in range(self.aquisition_parameters.src_positions.shape[0]):
            #     values.append(pre_values[j] * src_ti[j])
            
            # u_next = tf.sparse.add(u_next, tf.SparseTensor(pos, values, u_next.shape))
            # u_next = tf.sparse.add(u_next, tf.SparseTensor(pos, values[ti], u_next.shape))
            u_next = tf.sparse.add(u_next, src_values[ti])
            # print(tm.stop())

            u_prev = u
            u = u_next

            # print('Update:', end='  ')
            # tm.start()
            self.aquisition_parameters.update_rec_data(ti, u)
            # print(tm.stop())
            # print()

            if save_wavefield:
                u_history.append(u)
        
        # print('Main Loop:', tm2.stop())
        if save_wavefield:
            return u_history
        return []


    def __call__(self, *args, **kwargs):
        self.forward(*args, **kwargs)
    

    def adjoint(self):
        return WaveSolverOperator(self.aquisition_parameters.adjoint_parameters())