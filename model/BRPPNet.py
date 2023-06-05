"""YOLO_v3 Model Defined in Keras."""
import keras  
import tensorflow as tf
from keras import backend as K
from keras import layers as L
from keras.models import Model
from model.language_backbone import build_nlp_model
from model.attention_module import PSA
import model.utils as utils
import model.visual_backbone as V
import math

def simple_fusion(F_v, fq_word, name, dim=1024):
    """
    :param F_v: visual features (N,w,h,d)
    :param f_q: GRU output (N,T,d_q)
    :param dim: project dimensions default: 1024
    :return: F_m: fusion visual and textual features
    """
    
    out_size = K.int_shape(F_v)[1]
    batch, height, width, channels = K.int_shape(F_v)
    
    avg_x = L.Lambda(K.sum, arguments={'axis': 1})(fq_word)
    avg_x = L.Dense(dim, activation='linear')(avg_x)
    avg_x = L.advanced_activations.LeakyReLU(alpha=0.1)(L.normalization.BatchNormalization()(avg_x))
    avg_x = L.Lambda(utils.expand_and_tile, arguments={'outsize': out_size})(avg_x)
    
    out_y = L.Multiply()([F_v, avg_x])
    out_y = L.advanced_activations.LeakyReLU(alpha=0.1)(L.normalization.BatchNormalization()(out_y))
    
    return out_y
    
def MFFM(x, y, di=None, do=256):
    """
    concatenate upsample features x and selected features
    """
    
    if di is not None:
        x = V.DarknetConv2D_BN_Leaky(di, (3, 3))(x)
    x = L.UpSampling2D()(x)
    
    out = L.Concatenate()([x, y])
    out = V.DarknetConv2D_BN_Leaky(do, (1, 1))(out)
    
    key = L.Multiply()([out, x])
    key = V.DarknetConv2D_BN_Leaky(do//2, [1, 1])(key)
    key = V.DarknetConv2D_BN_Leaky(do, [3, 3])(key)
    out = L.Add() ([out, key])
    
    return out

def res_simple_fusion(F_v, fq_word, name, dim=1024):
    """
    :param F_v: visual features (N,w,h,d)
    :param f_q: GRU output (N,T,d_q)
    :param dim: project dimensions default: 1024
    :return: F_m: fusion visual and textual features
    """
    out_size = K.int_shape(F_v)[1]
    F_v_proj = V.darknet_resblock(F_v, dim // 2)
    
    avg_x = L.Lambda(K.sum, arguments={'axis': 1})(fq_word)
    avg_x = L.Dense(dim, activation='linear')(avg_x)
    avg_x = L.advanced_activations.LeakyReLU(alpha=0.1)(L.normalization.BatchNormalization()(avg_x))
    avg_x = L.Lambda(utils.expand_and_tile, arguments={'outsize': out_size})(avg_x)
    
    out_y = L.Multiply()([F_v_proj, avg_x])
    out_y = L.advanced_activations.LeakyReLU(alpha=0.1)(L.normalization.BatchNormalization()(out_y))
    
    return out_y
    
def BRPPNet_postproc(tf_out, config):
    # upsample the output of mdsfnet to get the same shape of ground-truth
    tf_out = V.DarknetConv2D_BN_Leaky(config.hidden_dim, [3, 3])(tf_out)
    tf_out = V.DarknetConv2D_BN_Leaky(config.hidden_dim, [3, 3])(tf_out)
    tf_out = V.DarknetConv2D_BN_Leaky(config.hidden_dim, [3, 3])(tf_out)
    
    if config.seg_out_stride <= 4:
    
        tf_out = L.UpSampling2D()(tf_out)
        tf_out = V.DarknetConv2D_BN_Leaky(config.hidden_dim, [3, 3])(tf_out)
    
    if config.seg_out_stride <= 2:
    
        tf_out = L.UpSampling2D()(tf_out)
        tf_out = V.DarknetConv2D_BN_Leaky(config.hidden_dim, [3, 3])(tf_out)
        
    tf_out = V.DarknetConv2D(1, [3, 3])(tf_out)

    return tf_out
    
def BRPPNet(Fv, q_input, config):
    # use bi-GRU and word-attention to get textual features
    fq_word = build_nlp_model(q_input=q_input, rnn_dim=config.rnn_hidden_size, dropout=config.rnn_drop_out)
    print(fq_word)
    
    # SEMI-ENCODER
    # (52,52,256)
    Fv_3 = V.darknet_resblock(Fv[2], K.int_shape(Fv[2])[3] // 2)
    fusion_3 = simple_fusion(Fv_3, fq_word, name='semi-encoder3', dim=K.int_shape(Fv_3)[3])
    
    # (26,26,512)
    Fv_2 = V.darknet_resblock(Fv[1], K.int_shape(Fv[1])[3] // 2)
    fusion_2 = simple_fusion(Fv_2, fq_word, name='semi-encoder2', dim=K.int_shape(Fv_2)[3])
    
    # (13,13,1024)
    Fv_1 = V.darknet_resblock(Fv[0], K.int_shape(Fv[0])[3] // 2)
    fusion_1 = simple_fusion(Fv_1, fq_word, name='semi-encoder1', dim=K.int_shape(Fv_1)[3])
    
    # DECODER.
    # (26,26,512)
    concat_2 = MFFM(fusion_1, fusion_2, di=512, do=512)
    
    # (52,52,256)
    concat_3 = MFFM(concat_2, fusion_3, di=256, do=256)
    
    # MLEM Upsample prediction to ground-truth
    select_3 = res_simple_fusion(concat_3, fq_word, name='decoder4', dim=K.int_shape(concat_3)[3])
    select_3 = PSA(select_3, concat_3)
    select_3 = L.Concatenate()([concat_3, select_3])
    select_3 = aspp_decoder(select_3, config.hidden_dim)
    mask_out = BRPPNet_postproc(select_3, config)

    return mask_out
    
def BRPPNet_body(inputs, q_input, config):
    """
    :param inputs:  image
    :param q_input:  word embeding
    :return:  regresion , attention map
    """
    """Create Multi-Modal YOLO_V3 model CNN body in Keras."""
    
    darknet = Model(inputs, V.darknet_body(inputs))
    # backbone visual features Fv[0]~Fv[2] (13,13,1024) (26,26,512) (52,52,256)
    Fv = [darknet.output, darknet.layers[152].output, darknet.layers[92].output]
    mask_out = BRPPNet(Fv, q_input, config)

    return Model([inputs, q_input], [mask_out])

def yolo_loss(args, batch_size, print_loss=False):
    mask_out =args[0]
    mask_gt = args[1]

    loss1 = 0
    m = K.shape(mask_out)[0]  # batch size, tensor
    mf = K.cast(m, K.dtype(mask_out))
    
    # bce loss
    mask_loss = K.binary_crossentropy(mask_gt, mask_out, from_logits=True)
    mask_loss = K.sum(mask_loss) / mf
    loss1 += mask_loss
    
    loss2 = 0
    bbce_loss = balanced_binary_cross_entropy(mask_out, mask_gt, mf)
    loss2 += bbce_loss

    loss3 = 0
    weight_loss = weighed_logistic_loss(mask_out, mask_gt, mf)
    loss3 += weight_loss
    
    if print_loss:
        loss = tf.Print(loss2, ['mask: ', loss1])

    return K.expand_dims(loss2, axis=0)

def balanced_binary_cross_entropy(pred, mask, batch, epsilon1=1.0, epsilon2 = -0.4, average=False):
    
    # Apply different weights to loss of positive samples and negative samples
    # Positive samples have the gradient weight of 1.0, while negative samples have the gradient weight of -0.4
    
    # Classification loss as the average or the sum of balanced per-score loss
    sig_pred = tf.nn.sigmoid(pred)
    pos_t = epsilon1 * (1 - mask * sig_pred)
    neg_t = epsilon2 * (1 - mask) * (1 - sig_pred)
    BCE = K.binary_crossentropy(mask, pred, from_logits=True)
    
    if average is True:
    
      BBCE = K.mean((BCE + (neg_t+pos_t)))
      
    else:
    
      BBCE = K.sum((BCE + (neg_t+pos_t)))
    
    bbce_loss = BBCE / batch
    
    return bbce_loss
    
def weighed_logistic_loss(scores, labels, mf, pos_loss_mult=3.0, neg_loss_mult=1.0):
    # Apply different weights to loss of positive samples and negative samples
    # positive samples have label 1 while negative samples have label 0
    loss_mult = tf.add(tf.multiply(labels, pos_loss_mult-neg_loss_mult), neg_loss_mult)
    #scores = scores + 1e-3
    # Classification loss as the average of weighed per-score loss
    cls_loss = K.sum(L.Multiply()([K.binary_crossentropy(labels, scores, from_logits=True), loss_mult]))
    cls_loss = cls_loss / mf 

    return cls_loss
    
# Atrous Spatial Pyramid Pooling module
def Aspp(tensor, output_channels=256, name_prefix="1"):
    '''atrous spatial pyramid pooling'''
    dims = K.int_shape(tensor)

    y_pool = L.AveragePooling2D(pool_size=(
        dims[1], dims[2]), name=name_prefix + 'average_pooling')(tensor)
    y_pool = L.Conv2D(filters=256, kernel_size=1, padding='same',
                      kernel_initializer='he_normal', name=name_prefix + 'pool_1x1conv2d', use_bias=False)(y_pool)
    y_pool = L.BatchNormalization(name=f'bn_1' + name_prefix)(y_pool)
    y_pool = L.Activation('relu', name=f'relu_1' + name_prefix)(y_pool)
    # y_pool = L.UpSampling2D((dims[1], dims[2]), 'bilinear')(y_pool)
    y_pool = L.Lambda(lambda x: tf.image.resize_bilinear(x, size=dims[1:3]))(y_pool)

    y_1 = L.Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same',
                   kernel_initializer='he_normal', name=name_prefix + 'ASPP_conv2d_d1', use_bias=False)(tensor)
    y_1 = L.BatchNormalization(name=f'bn_2' + name_prefix)(y_1)
    y_1 = L.Activation('relu', name=f'relu_2' + name_prefix)(y_1)

    y_6 = L.Conv2D(filters=256, kernel_size=3, dilation_rate=6, padding='same',
                   kernel_initializer='he_normal', name=name_prefix + 'ASPP_conv2d_d6', use_bias=False)(tensor)
    y_6 = L.BatchNormalization(name=f'bn_3' + name_prefix)(y_6)
    y_6 = L.Activation('relu', name=f'relu_3' + name_prefix)(y_6)

    y_12 = L.Conv2D(filters=256, kernel_size=3, dilation_rate=12, padding='same',
                    kernel_initializer='he_normal', name=name_prefix + 'ASPP_conv2d_d12', use_bias=False)(tensor)
    y_12 = L.BatchNormalization(name=f'bn_4' + name_prefix)(y_12)
    y_12 = L.Activation('relu', name=f'relu_4' + name_prefix)(y_12)

    y_18 = L.Conv2D(filters=256, kernel_size=3, dilation_rate=18, padding='same',
                    kernel_initializer='he_normal', name=name_prefix + 'ASPP_conv2d_d18', use_bias=False)(tensor)
    y_18 = L.BatchNormalization(name=f'bn_5' + name_prefix)(y_18)
    y_18 = L.Activation('relu', name=f'relu_5' + name_prefix)(y_18)

    y = L.concatenate([y_pool, y_1, y_6, y_12, y_18], name=name_prefix + 'ASPP_concat')

    y = L.Conv2D(filters=output_channels, kernel_size=1, dilation_rate=1, padding='same',
                 kernel_initializer='he_normal', name=name_prefix + 'ASPP_conv2d_final', use_bias=False)(y)
    y = L.BatchNormalization(name=f'bn_final' + name_prefix)(y)
    y = L.Activation('relu', name=f'relu_final' + name_prefix)(y)
    
    return y
    
def aspp_decoder(x, dim, output=False):
    shape=K.int_shape(x)
    b0 = V.DarknetConv2D_BN_Leaky(256, (1, 1), padding="same", use_bias=False)(x)


    b1=V.DarknetConv2D_BN_Leaky(256,(3,3), dilation_rate=(6, 6), padding="same", use_bias=False)(x)

    b2 = V.DarknetConv2D_BN_Leaky(256, (3, 3), dilation_rate=(12, 12), padding="same", use_bias=False)(x)

    b3 = V.DarknetConv2D_BN_Leaky(256, (3, 3), dilation_rate=(18, 18), padding="same", use_bias=False)(x)


    b4 = L.GlobalAveragePooling2D()(x)
    b4 = L.Lambda(K.expand_dims, arguments={'axis': 1})(b4)
    b4 = L.Lambda(K.expand_dims, arguments={'axis': 1})(b4)
    b4 = V.DarknetConv2D_BN_Leaky(256, (1, 1), padding="same", use_bias=False)(b4)
    b4 = L.Lambda(K.tile,arguments={'n':[1,shape[1],shape[2],1]})(b4)

    x = L.Concatenate()([b4, b0, b1, b2, b3])
    
    if output:
        x=V.DarknetConv2D(1,(1,1))(x)
        
    else:
    
        x=V.DarknetConv2D(dim,(1,1))(x)
        
    return x