import tensorflow as tf
from tensorflow.keras import layers, Model



def create_migration_map(nx, nz, name='migmap'):
    input_layer = layers.Input(shape=(nx,nz, 1))

    encoder_channels = [ 8, 16, 32, 64, 128 ]
    decoder_channels = [ 64, 32, 16, 8 ]

    x = input_layer
    skips = []
    # print('enc')
    for i, nc in enumerate(encoder_channels):
        x = _3x3Conv_BatchNorm_ReLU(nc, i!=0)(x)
        skips.insert(0, x)
        # print(x)
    skips.pop(0)


    # print('\ndec:')
    for i, nc in enumerate(decoder_channels):
        x = _2x2TransConv_Concat_3x3Conv_BatchNorm_ReLU(nc, skips[i])(x)
        # print(x)
    
    # output_layer = layers.Conv2D(1, (1,1), data_format='channels_first')(x)
    output_layer = layers.Conv2D(1, (1,1))(x)
    # output_layer = layers.BatchNormalization(axis=-1)(x)
    # output_layer = layers.ReLU()(x)

    
    return Model(inputs=input_layer, outputs=output_layer, name=name)



def _3x3Conv_BatchNorm_ReLU(nchannels, add_max_pooling=False):
    def f(x):
        if add_max_pooling:
            # x = layers.MaxPool2D((2,2), data_format='channels_first')(x)
            x = layers.MaxPool2D((2,2))(x)

        x = layers.Conv2D(nchannels, (3,3), padding='same')(x)
        # x = layers.BatchNormalization(axis=-1)(x)
        # x = layers.Dense
        # x = layers.ReLU()(x)
        x = layers.PReLU()(x)
        return x

    return f


def _2x2TransConv_Concat_3x3Conv_BatchNorm_ReLU(nchannels, concat_input):
    def f(x):
        # x = layers.Conv2DTranspose(nchannels, (2,2), data_format='channels_first', strides=2)(x)
        x = layers.Conv2DTranspose(nchannels, (2,2), strides=2)(x)
        # x = layers.UpSampling2D(size=(2,2),interpolation='bilinear')(x)
        # x = layers.Concatenate(axis=-1)([x, concat_input[:,:, :x.shape[2], :x.shape[3]]])
        x = layers.Concatenate(axis=-1)([x, concat_input])
        # x = layers.Concatenate(axis=0)([x, concat_input])
        x = _3x3Conv_BatchNorm_ReLU(nchannels)(x)

        return x

    return f