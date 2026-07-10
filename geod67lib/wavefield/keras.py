from geod67lib.plotting import plot_image

from .solver import WaveSolverOperator
from .aquisition import AquisitionParameters
from .model import VelocityModel

from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import PyDataset
import tensorflow as tf
import numpy as np



def pad_edge(x: tf.Tensor, s: int):
    paddings = tf.constant([[0, 0], [1, 1], [1,1], [0,0] ])
    for _ in range(s):
        x = tf.pad(x, paddings, mode='SYMMETRIC')
    return x


class WaveSolverLayer(Layer):
    def __init__(self, aquisition_parameters: AquisitionParameters, damp_mask: np.ndarray, 
                 water_mask: np.ndarray, spacing, shape, fd_order=2, **kwargs):
        
        self.op = WaveSolverOperator(aquisition_parameters)
        self.damp_mask = damp_mask
        self.water_mask = water_mask
        self.hx = spacing[0]
        self.hz = spacing[1]
        self.inner_shape = shape
        self.nbl = (damp_mask.shape[1] - shape[0])//2
        self.fd_order = fd_order
        super().__init__(**kwargs)
    
    
    # def call(self, vp_data):
    #     vp_data = vp_data * self.water_mask
    #     nbl = (self.damp_mask.shape[1] - vp_data.shape[1])//2
    #     vp_data = pad_edge(vp_data, nbl)
    #     # tf.debugging.check_numerics(vp_data, message=f'Checking vp')

    #     vp = VelocityModel(vp_data, self.hx, self.hz)

    #     dcalc = []
    #     op = self.op
    #     for isrc in range(self.src_positions.shape[0]):
    #         src_pos = self.src_positions[isrc]
    #         op.aquisition_parameters.reset()
    #         op.aquisition_parameters.src_positions[0] = src_pos
    #         op.forward(vp, self.damp_mask, fd_order=self.fd_order)
    #         # tf.debugging.check_numerics(op.aquisition_parameters.rec_data, message=f'Checking shot {isrc + 1}')
    #         dcalc.append(op.aquisition_parameters.rec_data)
    #     dcalc = tf.stack(dcalc)
    #     return dcalc

    def _while_loop_body(self, i, src_vp_data, dcalc):
        src_pos = tf.cast(src_vp_data[i][:2], tf.int32)
        vp_data = src_vp_data[i][2:]
        vp_data = tf.reshape(vp_data, (1, *self.inner_shape, 1))
        vp_data = vp_data * self.water_mask + 1.5 * (1 - self.water_mask)
        # try:
        #     plot_image(vp_data.numpy())
        # except:
        #     pass
        # nbl = (self.damp_mask.shape[1] - vp_data.shape[1])//2
        # tf.debugging.check_numerics(vp_data, message=f'Checking vp')
        vp_data = pad_edge(vp_data, self.nbl)
        # tf.debugging.check_numerics(vp_data, message=f'Checking vp')

        vp = VelocityModel(vp_data, self.hx, self.hz)
        # try:
        #     plot_image(vp.data.numpy())
        # except:
        #     pass

        op = self.op
        op.aquisition_parameters.reset()
        op.aquisition_parameters.src_positions = tf.cast(tf.reshape(src_pos, (1, *(src_pos.shape))), tf.int32)

        op.forward(vp, self.damp_mask, fd_order=self.fd_order)
        # tf.debugging.check_numerics(op.aquisition_parameters.rec_data, message=f'Checking shot {isrc + 1}')
        dcalc = dcalc.write(i, op.aquisition_parameters.rec_data)
        return [tf.add(i, 1), src_vp_data, dcalc]
    
    def call(self, src_vp_data):
        nbatch = tf.shape(src_vp_data)[0]
        dcalc = tf.TensorArray(element_shape = (self.op.aquisition_parameters.nrec, self.op.aquisition_parameters.nt), dtype=tf.float32, size=0, dynamic_size=True)
        
        i = tf.constant(0)

        cond = lambda i, _1, _2 : tf.less(i, nbatch)
        res = tf.while_loop(cond, self._while_loop_body, [i, src_vp_data, dcalc])
        dcalc = res[2]
        dcalc = dcalc.stack()

        return dcalc
    


class MigFWIDataset(PyDataset):
    def __init__(self, src_positions, zeta, dobs, batch_size=1, **kwargs):
        super().__init__(**kwargs)
        self.src_positions = 1*src_positions
        self.zeta = 1*zeta
        self.batch_size = batch_size
        self.dobs = dobs
    

    def __len__(self):
        return int(tf.math.ceil(self.src_positions.shape[0]/self.batch_size))
    
    
    def __getitem__(self, idx):
        base = idx * self.batch_size

        x_src = self.src_positions[base : base + self.batch_size]
        x_zeta = np.array([ self.zeta ] * self.batch_size).reshape((self.batch_size, *self.zeta.shape))
        
        y = self.dobs[base : base + self.batch_size]
        
        return ( x_src, x_zeta ), y
    

def MSECossineSimilarityLoss():
    cossim = tf.keras.losses.CosineSimilarity(axis=0)
    def mse_cosine_similiarity_loss(dcalc, dobs):
        loss = tf.reduce_mean((dcalc - dobs)**2)
        loss += cossim(dcalc, dobs)
        return loss
    return mse_cosine_similiarity_loss
