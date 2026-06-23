import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

def plot_image(data, cmap='jet', title=None, label=None, vmin=None, vmax=None):
    if len(data.shape) == 4:
        data = data[0, :, :, 0]

    plt.figure()

    plt.imshow(data.T, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label=label)

    if not(title is None or title.strip() == ''):
        plt.title(title)
    

    plt.show()


def plot_seismogram(rec_data, width, tn, scale_factor=10):
    if isinstance(rec_data, tf.Tensor):
        rec_data = rec_data.numpy()

    vmax = np.abs(rec_data).max()/scale_factor

    plt.figure(figsize=(10,8))
    plt.imshow(tf.transpose(rec_data), extent=[
        0, width, tn, 0
    ], cmap='grey', vmax=vmax, vmin=-vmax)

    plt.colorbar()