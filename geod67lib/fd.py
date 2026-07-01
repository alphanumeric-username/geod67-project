# Finite differences method

import numpy as np
import tensorflow as tf
from scipy.special import factorial


_cached_coeffs = {}
def taylor_coeffs(k):
    if not(k in _cached_coeffs):
        e = np.arange(2*k + 1)
        d = factorial(e)

        X = np.array([
            i**e/d for i in range(k, -k-1, -1)
        ], dtype=np.float32)

        W = np.linalg.inv(X)
        _cached_coeffs[k] = W
        
    return _cached_coeffs[k]


_cached_kernels = {}
def d(f, h, axis=0, fd_order=2, op_order=1, **kwargs):
    # cache_key = (fd_order, op_order, axis)

    # kernel = None
    # if not(cache_key in _cached_kernels):
    coeffs = taylor_coeffs(fd_order//2)[op_order]/h**op_order
    coeffs = np.flip(coeffs)
    n = coeffs.shape[0]
    kernel = np.zeros((n,n,1,1), dtype=np.float32)
    if axis == 0:
        kernel[:, n//2, 0, 0] = coeffs
    else:
        kernel[n//2, :, 0, 0] = coeffs
    kernel = tf.constant(kernel)
    #     _cached_kernels[cache_key] = kernel.numpy()
    # else:
    #     kernel = tf.constant(_cached_kernels[cache_key])

    out = tf.nn.conv2d(f, kernel, strides=1, padding='SAME')
    return out



def lap(f, hx, hz, **kwargs):
    """
    Second-order 2D laplacian operator
    """
    return d(f, hx, axis=0, op_order=2, **kwargs) + d(f, hz, axis=1, op_order=2, **kwargs)


def grad(f, hx, hz, **kwargs):
    d(f, hx, axis=0, **kwargs), d(f, hz, axis=1, **kwargs)



class DifferentialOperator():
    def __init__(self, fd_order, hx, hz):
        self.W = taylor_coeffs(fd_order//2)
        self.kernels = {}
        h = [hx, hz]
        for axis in [0, 1]:
            for op_order in range(self.W.shape[0]):
                key = (axis, op_order)

                coeffs = self.W[op_order]/h[axis]**op_order
                coeffs = np.flip(coeffs)
                n = coeffs.shape[0]
                kernel = np.zeros((n,n,1,1), dtype=np.float32)
                if axis == 0:
                    kernel[:, n//2, 0, 0] = coeffs
                else:
                    kernel[n//2, :, 0, 0] = coeffs
                kernel = tf.constant(kernel)
                
                self.kernels[key] = kernel
    

    @tf.function
    def d(self, f, axis=0, op_order=1):
        kernel = self.kernels[(axis, op_order)]
        out = tf.nn.conv2d(f, kernel, strides=1, padding='SAME')
        return out
    
    @tf.function
    def lap(self, f):
        return self.d(f, axis=0, op_order=2) + self.d(f, axis=1, op_order=2)

    @tf.function
    def grad(self, f):
        self.d(f, axis=0, op_order=1), self.d(f, axis=1, op_order=1)