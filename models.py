from tensorflow.keras.layers import Input, Dense, BatchNormalization, \
    Conv2DTranspose, Conv2D, LeakyReLU, ReLU, Reshape, Flatten, concatenate, GaussianNoise
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import L2
from tensorflow.keras.models import Model



regularizer = L2(2.5e-5)
initializer = RandomNormal(stddev=0.02)


def dense_block(units, input_layer, reshape_dim=None):
    dense = Dense(units,
                  kernel_regularizer=regularizer,
                  kernel_initializer=initializer)(input_layer)
    bn = BatchNormalization()(dense)
    if reshape_dim is not None:
        bn = Reshape(reshape_dim)(bn)
    out = ReLU()(bn)

    return out


def upconv_block(n_filter, filter_size, filter_stride, input_layer, bn=False):
    upconv = Conv2DTranspose(n_filter, filter_size,
                             strides=filter_stride,
                             kernel_initializer=initializer,
                             kernel_regularizer=regularizer,
                             padding='same')(input_layer)
    if bn:
        upconv = BatchNormalization()(upconv)
    out = ReLU()(upconv)

    return out


def conv_block(n_filter, filter_size, filter_stride, input_layer):
    conv = Conv2D(n_filter, filter_size,
                  strides=filter_stride,
                  kernel_initializer=initializer,
                  kernel_regularizer=regularizer)(input_layer)
    out = LeakyReLU(0.2)(conv)

    return out


def encoder(in_sh=(28, 28, 1), out_dim=40):
    input_layer = Input(shape=in_sh)
    conv1 = conv_block(64, (4, 4), (2, 2), input_layer)
    conv2 = conv_block(128, (4, 4), (2, 2), conv1)
    flat = Flatten()(conv2)
    dense = Dense(1024,
                  kernel_regularizer=regularizer,
                  kernel_initializer=initializer,
                  activation=LeakyReLU(0.2))(flat)
    out = Dense(out_dim,
                kernel_regularizer=regularizer,
                kernel_initializer=initializer)(dense)

    model = Model(inputs=input_layer, outputs=out)

    return model

def aae_generator(in_sh=(40,)):
    input_layer = Input(shape=in_sh)
    dense1 = dense_block(1024, input_layer)
    dense2 = dense_block(7 * 7 * 128, dense1, reshape_dim=(7, 7, 128))
    upconv1 = upconv_block(64, (4, 4), (2, 2), dense2, bn=True)
    out = Conv2DTranspose(1, (4, 4), (2, 2),
                          kernel_initializer=initializer,
                          kernel_regularizer=regularizer,
                          activation='sigmoid', padding='same')(upconv1)

    model = Model(inputs=input_layer, outputs=out)

    return model

def aae_discriminator(in_sh=(40,)):
    input_layer = Input(shape=in_sh)

    flat = Flatten()(input_layer)
    dense = Dense(1024,
                  kernel_regularizer=regularizer,
                  kernel_initializer=initializer,
                  activation=LeakyReLU(0.2))(flat)
    out = Dense(1,
                kernel_regularizer=regularizer,
                kernel_initializer=initializer)(dense)

    model = Model(inputs=input_layer, outputs=out)

    return model

