import tensorflow as tf


def create_normalization_transforms(vp_max, vp_min):
    a = vp_min
    b = vp_max

    forward = lambda vp : tf.math.log((vp - a)/(b - vp))
    inverse = lambda m : (a + b * tf.math.exp(m))/(1 + tf.math.exp(m))

    return forward, inverse