from .solver import WaveSolverOperator
from .aquisition import AquisitionParameters
from .model import VelocityModel

from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np



def pad_edge(x: tf.Tensor, s: int):
    paddings = tf.constant([[0, 0], [1, 1], [1,1], [0,0] ])
    for _ in range(s):
        x = tf.pad(x, paddings, mode='REFLECT')
    return x


class WaveSolverLayer(Layer):
    def __init__(self, aquisition_parameters: AquisitionParameters, damp_mask: np.ndarray, 
                 src_positions: np.ndarray, spacing, fd_order=2, **kwargs):
        
        self.op = WaveSolverOperator(aquisition_parameters)
        self.damp_mask = damp_mask
        self.src_positions = src_positions
        self.hx = spacing[0]
        self.hz = spacing[1]
        self.fd_order = fd_order
        super().__init__(**kwargs)
    
    
    def call(self, vp_data):
        nbl = (self.damp_mask.shape[1] - vp_data.shape[1])//2
        vp_data = pad_edge(vp_data, nbl)

        vp = VelocityModel(vp_data, self.hx, self.hz)
        dcalc = []
        op = self.op
        for isrc in range(self.src_positions.shape[0]):
            src_pos = self.src_positions[isrc]
            op.aquisition_parameters.reset()
            op.aquisition_parameters.src_positions[0] = src_pos
            op.forward(vp, self.damp_mask, fd_order=self.fd_order)
            dcalc.append(op.aquisition_parameters.rec_data)
        dcalc = tf.stack(dcalc)
        return dcalc
    

