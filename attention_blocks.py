import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Reshape, Activation, multiply, Lambda, Concatenate, Conv2D, add
import keras.backend as K

###########################
##                       ##
##  Blocos de attention  ##
##                       ##
###########################

def channel_attention(input_feature, ratio=8):
    channel = K.int_shape(input_feature)[-1]
    shared_layer_one = Dense(
        channel // ratio,
        activation='relu',
        kernel_initializer='he_normal',
        use_bias=True,
        bias_initializer='zeros',
        name='cbam_1'
    )
    shared_layer_two = Dense(
        channel,
        kernel_initializer='he_normal',
        use_bias=True,
        bias_initializer='zeros',
        name='cbam_2'
    )

    avg_pool = GlobalAveragePooling2D(name='cbam_avg_pool')(input_feature)
    avg_pool = Reshape((1, 1, channel), name='cbam_reshape_avg')(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPooling2D(name='cbam_max_pool')(input_feature)
    max_pool = Reshape((1, 1, channel), name='cbam_reshape_max')(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = add([avg_pool, max_pool], name='cbam_add')
    cbam_feature = Activation('sigmoid', name='cbam_activation')(cbam_feature)

    return multiply([input_feature, cbam_feature], name='cbam_multiply')


def spatial_attention(input_feature):
    kernel_size = 7

    if K.image_data_format() == 'channels_first':
        channel = 1
    else:
        channel = -1

    avg_pool = Lambda(lambda x: K.mean(x, axis=channel, keepdims=True))(input_feature)
    max_pool = Lambda(lambda x: K.max(x, axis=channel, keepdims=True))(input_feature)
    concat = Concatenate(axis=channel)([avg_pool, max_pool])
    cbam_feature = Conv2D(
        filters=1,
        kernel_size=kernel_size,
        strides=1,
        padding='same',
        activation='sigmoid',
        kernel_initializer='he_normal',
        use_bias=False
    )(concat)

    return multiply([input_feature, cbam_feature])


def cbam_block(cbam_feature, ratio=8):
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature


def squeeze_excite_block(input_tensor, ratio=16):
    init = input_tensor
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D(name='se_pooling')(init)
    se = Reshape(se_shape, name='se_reshape')(se)
    se = Dense(filters // ratio, activation='relu', use_bias=False, name='se_dense_1')(se)
    se = Dense(filters, activation='sigmoid', use_bias=False, name='se_dense_2')(se)

    x = multiply([init, se], name='se_multiply')
    return x
