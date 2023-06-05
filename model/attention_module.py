from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, multiply, Concatenate
from keras.layers import Conv2D, Reshape, Softmax, Lambda, Activation, Multiply, GlobalAvgPool2D, Add, Permute
import tensorflow as tf
from keras import layers as L
from keras import backend as K
from keras_layer_normalization import LayerNormalization

def PSA(x, y, mode='p'):
    context_channel = spatial_pool(x, y, mode)
    if mode == 'p':
        context_spatial = channel_pool(x, y)
        out = Add()([context_spatial, context_channel])
    elif mode == 's':
        out = channel_pool(context_channel, y)
    else:
        out = x
    return out
    
def spatial_pool(x, y, mode='p', ratio=4):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    if channel_axis == -1:
        batch, height, width, channels = K.int_shape(x)
        assert channels % 2 == 0
        channel = channels // 2
        input_x = Conv2D(channel, kernel_size=1, strides=1, padding='same', use_bias=False,
                         kernel_initializer='he_normal')(x)
        input_x = Reshape((width * height, channel))(input_x)

        context_mask = Conv2D(1, kernel_size=1, strides=1, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
        context_mask = Reshape((width * height, 1))(context_mask)
        context_mask = Softmax(axis=1)(context_mask)
        context = Lambda(lambda x: tf.matmul(x[0], x[1], transpose_a=True))([input_x, context_mask])
        context = Permute((2, 1))(context)
        context = Reshape((1, 1, channel))(context)

    else: 
        batch, channels, height, width = K.int_shape(x)
        assert channels % 2 == 0
        channel = channels // 2
        input_x = Conv2D(channel, kernel_size=1, strides=1, padding='same', use_bias=False,
                         kernel_initializer='he_normal')(x)
        input_x = Reshape((channel, width * height))(input_x)

        context_mask = Conv2D(1, kernel_size=1, strides=1, padding='same', use_bias=False,
                              kernel_initializer='he_normal')(x)
        context_mask = Reshape((width * height, 1))(context_mask)
        context_mask = Softmax(axis=1)(context_mask)
        context = Lambda(lambda x: tf.matmul(x[0], x[1]))([input_x, context_mask])
        context = Reshape((channel, 1, 1))(context)

    if mode == 'p':
        context = Conv2D(channels, kernel_size=1, strides=1, padding='same')(context)
    else:
        context = Conv2D(channel // ratio, kernel_size=1, strides=1, padding='same')(context)
        context = LayerNormalization()(context)  # pip install keras-layer-normalization
        context = Conv2D(channels, kernel_size=1, strides=1, padding='same')(context)

    mask_ch = Activation('sigmoid')(context)
    
    return Multiply()([y, mask_ch])

def channel_pool(x, y):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    if channel_axis == -1:
        batch, height, width, channels = K.int_shape(x)
        assert channels % 2 == 0
        channel = channels // 2
        g_x = Conv2D(channel, kernel_size=1, strides=1, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
        avg_x = GlobalAvgPool2D()(g_x)
        avg_x = Softmax()(avg_x)
        avg_x = Reshape((channel, 1))(avg_x)

        theta_x = Conv2D(channel, kernel_size=1, strides=1, padding='same', use_bias=False,
                         kernel_initializer='he_normal')(x)
        theta_x = Reshape((height * width, channel))(theta_x)
        context = Lambda(lambda x: tf.matmul(x[0], x[1]))([theta_x, avg_x])
        context = Reshape((height * width,))(context)
        mask_sp = Activation('sigmoid')(context)
        mask_sp = Reshape((height, width, 1))(mask_sp)
    else:
        batch, channels, height, width = K.int_shape(x)
        assert channels % 2 == 0
        channel = channels // 2
        g_x = Conv2D(channel, kernel_size=1, strides=1, padding='same', use_bias=False, kernel_initializer='he_normal')(
            x)
        avg_x = GlobalAvgPool2D()(g_x)
        avg_x = Softmax()(avg_x)
        avg_x = Reshape((1, channel))(avg_x)

        theta_x = Conv2D(channel, kernel_size=1, strides=1, padding='same', use_bias=False,
                         kernel_initializer='he_normal')(x)
        theta_x = Reshape((channel, height * width))(theta_x)
        context = Lambda(lambda x: tf.matmul(x[0], x[1]))([avg_x, theta_x])
        context = Reshape((height * width,))(context)
        mask_sp = Activation('sigmoid')(context)
        mask_sp = Reshape((1, height, width))(mask_sp)
        
    return Multiply()([y, mask_sp])
