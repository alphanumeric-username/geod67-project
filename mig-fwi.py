import argparse
import datetime
import json
import sys

import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import clear_output


import numpy as np

import tensorflow as tf
from tensorflow.keras import  layers, Model

from geod67lib.unet import create_migration_map
from geod67lib.normalization import create_normalization_transforms
from geod67lib.plotting import plot_image, plot_seismogram
from geod67lib.wavefield import WaveSolverLayer, AquisitionParameters, generate_damp_mask

import time

from geod67lib.wavefield.keras import MigFWIDataset, MSECossineSimilarityLoss


def main(argv):
    t0 = time.time()
    args = parse_args(argv)
    learning_rate = args.learning_rate


    shape = (128, 128)
    spacing = (20, 20)
    nbl = 40

    width, height = (shape[0] - 1)*spacing[0], (shape[1] - 1)*spacing[1]


    vp_true = np.fromfile(f'./data/marmo_vp_nx{shape[0]}nz{shape[1]}dx{spacing[0]}dz{spacing[1]}.bin', dtype=np.float32).reshape((1, shape[0], shape[1], 1))
    vp_0 = np.fromfile(f'./data/vp0_nx{shape[0]}nz{shape[1]}dx{spacing[0]}dz{spacing[1]}.bin', dtype=np.float32).reshape((1, shape[0], shape[1], 1))
    zeta = np.fromfile(f'./data/mig_nx{shape[0]}nz{shape[1]}dx{spacing[0]}dz{spacing[1]}.bin', dtype=np.float32).reshape((1, shape[0], shape[1], 1))


    vpmin = vp_true.min()*.9
    vpmax = vp_true.max()*1.1


    normalize, denormalize = create_normalization_transforms(vpmin, vpmax)
    m_0 = normalize(tf.constant(vp_0))

    migmap = create_migration_map(*shape)

    zeta_in = tf.constant(zeta)
    dm = migmap(zeta_in)


    ap = AquisitionParameters.load('./data/aquisition-params.json')

    nrec = ap.nrec
    nsrc = ap.nsrc
    nt = ap.nt

    src_positions = ap.src_positions
    ap.src_positions = np.zeros((1, 2))


    vp0_pad = np.pad(vp_0[0, :, :, 0], nbl, mode='edge')
    vp0_pad = vp0_pad.reshape((1, *vp0_pad.shape, 1))
    damp_mask = generate_damp_mask(vp0_pad, nbl)
    water_mask = np.fromfile(f'data/watermask_nx{shape[0]}nz{shape[1]}dx{spacing[0]}dz{spacing[1]}.bin', dtype=np.int8).reshape(1, *shape, 1)


    dobs = np.fromfile(f'./data/dobs_nx{shape[0]}nz{shape[1]}dx{spacing[0]}dz{spacing[1]}.bin', dtype=np.float32).reshape(nsrc, nrec, nt)


    input_src = layers.Input(shape=(2,), name='in_src')
    input_zeta = layers.Input(shape=(*shape, 1), name='in_zeta')

    dm = migmap(input_zeta)
    m = layers.Add()([dm, m_0])
    vp_full = layers.Lambda(denormalize)(m)
    vp_full =layers.Reshape((-1,))(vp_full)
    src_pos = layers.Reshape((-1,))(input_src)
    src_vp = layers.Concatenate(axis=-1)([src_pos, vp_full])
    wsl = WaveSolverLayer(ap, damp_mask, water_mask, spacing=spacing, shape=shape, fd_order=4,)
    out = wsl(src_vp)

    model = Model(inputs=[input_src, input_zeta],outputs=out,name='train_net')


    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        # loss=MSECossineSimilarityLoss(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate,),
        metrics=[
            tf.keras.metrics.MeanSquaredError()
        ]
    )

    if not(args.checkpoint is None):
        model.load_weights(args.checkpoint)
    


    cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        monitor='loss',
        filepath=args.weightfile,
        save_weights_only=True,
        save_best_only=True
    )

    cb_earlystop = tf.keras.callbacks.EarlyStopping(
        monitor='mean_squared_error',
        patience=100
    )


    dataset = MigFWIDataset(src_positions, zeta.reshape((zeta.shape[1], zeta.shape[2], 1)), dobs, batch_size=5)


    history = model.fit(
        dataset,
        batch_size=dataset.batch_size,
        # verbose=2,
        epochs=1500,
        # epochs=2000,
        callbacks=[ cb_checkpoint, cb_earlystop ]
    )

    dm = migmap(zeta)

    # denormalize(dm + m_0).numpy().tofile(f'./data/vp_inv_nx{shape[0]}nz{shape[1]}dx{spacing[0]}dz{spacing[1]}.bin')
    denormalize(dm + m_0).numpy().tofile(args.outfile)

    with open(f'{args.outfile}.history.json', 'w+') as fout:
        json.dump(history.history, fout, indent=4)

    print(f'Finished in {datetime.timedelta(seconds=time.time() - t0)}')
    print('Done')

    return 0




def parse_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning-rate', '-l',  type=float, default=0.001)
    parser.add_argument('--outfile', '-o',  type=str, default='./data/out.bin')
    parser.add_argument('--weightfile', '-w',  type=str, default='./data/migmap.weights.h5')
    parser.add_argument('--checkpoint', '-c',  type=str, default=None)

    return parser.parse_args(argv)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))