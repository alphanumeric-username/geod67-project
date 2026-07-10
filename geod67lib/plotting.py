import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

def plot_image(data, cmap='jet', title=None, label=None, vmin=None, vmax=None, width=None, height=None, hide_ticks=False, hide_cbar=False, figsize=None, xlabel=None, ylabel=None):
    if len(data.shape) == 4:
        data = data[0, :, :, 0]

    plt.figure(figsize=figsize)

    if not(width is None and height is None):
        extent = [
            0, width, height, 0
        ]
    else:
        extent = None

    plt.imshow(data.T, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='spline36', extent=extent)
    if not hide_cbar:
        plt.colorbar(label=label)

    if hide_ticks:
        plt.tick_params(
            axis='both',
            which='both',
            left=False,
            right=False,
            labelleft=False,
            bottom=False,
            top=False,
            labelbottom=False
        )

    if not(title is None or title.strip() == ''):
        plt.title(title)

    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    

    plt.show()


def plot_seismogram(rec_data, width, tn, scale_factor=10, clean_plot=False):
    if isinstance(rec_data, tf.Tensor):
        rec_data = rec_data.numpy()

    vmax = np.abs(rec_data).max()/scale_factor

    plt.figure(figsize=(10,8))
    plt.imshow(tf.transpose(rec_data), extent=[
        0, width, tn, 0
    ], cmap='grey', vmax=vmax, vmin=-vmax)

    if clean_plot:
        plt.tick_params(
            axis='both',
            which='both',
            left=False,
            right=False,
            labelleft=False,
            bottom=False,
            top=False,
            labelbottom=False
        )
    else:
        plt.colorbar()