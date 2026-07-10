import numpy as np
import tensorflow as tf

class VelocityModel:
    def __init__(self, data, hx, hz, nbl = 0):
        self.data = data
        self.hx = hx
        self.hz = hz
        self.nbl = nbl
        # self.damp_mask: tf.Tensor = None
    
    @property
    def width(self):
        return (self.data.shape[1] - 1) * self.hx
    
    @property
    def height(self):
        return (self.data.shape[2] - 1) * self.hz
    
    @property
    def inner_width(self):
        return (self.data.shape[1] - 1) * self.hx - 2*self.nbl * self.hx
    
    @property
    def inner_height(self):
        return (self.data.shape[2] - 1) * self.hz - 2*self.nbl * self.hz
    
    @property
    def shape(self):
        # return self.data.shape[1:3]
        return self.data.shape
    
    @property
    def sliced_data(self):
        return self.data[0,:,:,0]
    
    @property
    def inner_data(self):
        if self.nbl == 0:
            return self.data
        return self.data[:, self.nbl:-self.nbl, self.nbl:-self.nbl, :]

    @property
    def sliced_inner_data(self):
        if self.nbl == 0:
            return self.sliced_data
        return self.data[0, self.nbl:-self.nbl, self.nbl:-self.nbl, 0]