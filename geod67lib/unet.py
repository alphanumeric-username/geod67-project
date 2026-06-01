import tensorflow as tf
from tensorflow.keras import layers, models


# class Unet():
#     def __init__(self, nx, nz):
#         self.nx = nx
#         self.nz = nz

#         self._model = self._create_net()



#     def _create_net(self):

#         model = models.Sequential()

#         encoder = [

#         ]


#         pass


def _3x3ConvBatchRelu(nchannels, input_shape=None, add_max_pooling=False):
    l = [
        layers.Conv2D(nchannels, (3,3), input_shape=input_shape, data_format='channels_first'),
        layers.BatchNormalization(axis=1),
        layers.ReLU(),
    ]

    if add_max_pooling:
        l.insert(0, layers.MaxPooling2D((2,2)))

    return l


def create_mig_fwi_unet(nx, nz):
    unet = models.Sequential()
    
    encoder_channels = [ 8, 16, 32, 64, 128 ]
    decoder_channels = [ 64, 32, 16, 8 ]

    encoder = [ 
        l for i, nc in enumerate(encoder_channels)
          for l in _3x3ConvBatchRelu(nc, (nx, nz), add_max_pooling=i==0) 
    ]

    # encoder = [
    #     *_3x3ConvBatchRelu(8, (nx, nz)),
    #     *_3x3ConvBatchRelu(16, add_max_pooling=True),
    #     *_3x3ConvBatchRelu(32, add_max_pooling=True),
    #     *_3x3ConvBatchRelu(64, add_max_pooling=True),
    #     *_3x3ConvBatchRelu(128, add_max_pooling=True),
    # ]

    decoder = [
        
    ]

    return unet
    