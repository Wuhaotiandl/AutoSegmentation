import keras
from keras.layers import SeparableConvolution2D
from keras.layers import Dense, Dropout, Flatten,Input
from keras.layers import Conv2D, MaxPooling2D, Conv3D, MaxPooling3D, Add, Conv3DTranspose
from keras.optimizers import SGD,Adam
from keras.layers import BatchNormalization
from keras.layers import concatenate
from keras.layers.core import Flatten
from keras.models import Model
import json
from keras.layers import UpSampling2D

"""
    liver的3d勾画模型
"""
def seg_liver3d_8():
    inputs = Input(shape=(8, 256, 256, 1))
    concatenate_1 = Liver3DUnit(inputs, 4)
    max_pooling3d_1 = MaxPooling3D(pool_size=(1, 4, 4), strides=(1, 4, 4), padding='valid')(concatenate_1)
    conv3d_5 = Conv3D(16, (1, 4, 4), (1, 4, 4), padding='valid', activation='selu')(concatenate_1)
    concatenate_2 = Liver3DUnit(conv3d_5, 4)
    add_1 = Add()([concatenate_2, max_pooling3d_1])

    concatenate_3 = Liver3DUnit(add_1, 4)
    concatenate_4 = Liver3DUnit(concatenate_3, 4)

    add_2 = Add()([concatenate_4, add_1])
    concatenate_5 = Liver3DUnit(add_2, 8)
    max_pooling3d_2 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid')(concatenate_5)
    conv3d_22 = Conv3D(32, (1, 2, 2), (1, 2, 2), padding='valid', activation='selu')(concatenate_5)
    concatenate_6 = Liver3DUnit(conv3d_22, 8)
    add_3 = Add()([concatenate_6, max_pooling3d_2])

    concatenate_7 = Liver3DUnit(add_3, 8)
    concatenate_8 = Liver3DUnit(concatenate_7, 8)
    add_4 = Add()([add_3, concatenate_8])

    conv3d_transpose_1 = Conv3DTranspose(32, (1, 2, 2), (1, 2, 2), padding='valid', activation='selu')(add_4)
    concatenate_9 = concatenate([conv3d_transpose_1, add_2], axis=4)

    concatenate_10 = Liver3DUnit(concatenate_9, 8)
    concatenate_11 = Liver3DUnit(concatenate_10, 12)
    add_5 = Add()([concatenate_11, concatenate_9])

    concatenate_12 = Liver3DUnit(add_5, 12)
    concatenate_13 = Liver3DUnit(concatenate_12, 12)
    add_6 = Add()([concatenate_13, add_5])

    conv3d_transpose_2 = Conv3DTranspose(16, (1, 4, 4), (1, 4, 4), padding='valid', activation='selu')(add_6)
    concatenate_14 = concatenate([conv3d_transpose_2, concatenate_1], axis=4)

    concatenate_15 = Liver3DUnit(concatenate_14, 4)
    concatenate_16 = Liver3DUnit(concatenate_15, 8)
    add_7 = Add()([concatenate_16, concatenate_14])

    concatenate_17 = Liver3DUnit(add_7, 8)
    concatenate_18 = Liver3DUnit(concatenate_17, 8)
    add_8 = Add()([concatenate_18, add_7])

    conv3d_67 = Conv3D(16, (1, 1, 1), (1, 1, 1), padding='saame', activation='selu')(add_8)
    concatenate_19 = Liver3DUnit(conv3d_67, 4)
    concatenate_20 = Liver3DUnit(concatenate_19, 4)
    add_9 = Add()([concatenate_20, conv3d_67])

    concatenate_21 = Liver3DUnit(add_9, 4)
    concatenate_22 = Liver3DUnit(concatenate_21, 4)
    add_10 = Add()([concatenate_22, add_9])

    conv3d_84 = Conv3D(8, (1, 1, 1), (1, 1, 1), padding='same', activation='selu')(add_10)
    out = Conv3D(1, (1, 1, 1), (1, 1, 1), padding='valid', activation='sigmoid')(conv3d_84)
    return out









def Liver3DUnit(model, filters):
    Conv3D_1 = Conv3D(filters, (3, 3, 3), (1, 1, 1), padding='same', activation='selu', dilation_rate=(1, 1, 1))(model)
    Conv3D_2 = Conv3D(filters, (3, 3, 3), (1, 1, 1), padding='same', activation='selu', dilation_rate=(1, 2, 2))(model)
    Conv3D_3 = Conv3D(filters, (3, 3, 3), (1, 1, 1), padding='same', activation='selu', dilation_rate=(1, 3, 3))(model)
    Conv3D_4 = Conv3D(filters, (3, 3, 3), (1, 1, 1), padding='same', activation='selu', dilation_rate=(1, 5, 5))(model)
    Concatenate = concatenate([Conv3D_1, Conv3D_2, Conv3D_3, Conv3D_4], axis=4)
    return Concatenate

