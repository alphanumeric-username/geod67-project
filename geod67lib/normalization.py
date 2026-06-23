import tensorflow as tf


def create_normalization_transforms(vp_min, vp_max):
    a = vp_min
    b = vp_max

    normalize = lambda vp : tf.math.log((vp - a)/(b - vp))
    denormalize = lambda m : (a + b * tf.math.exp(m))/(1 + tf.math.exp(m))

    return normalize, denormalize