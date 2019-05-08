from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.layers import BatchNormalization
from keras.layers import concatenate
from keras.layers.core import Flatten
from keras.models import Model
import tensorflow as tf
from keras import backend as k

def sbss_net():
    inputs = Input(shape=(32, 32, 1))
    sbssModel = Conv2D(64, (5, 5), padding='same', activation='tanh')(inputs)
    sbssModel = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(sbssModel)
    sbssModel = Conv2D(128, (3, 3), padding='same', activation='tanh')(sbssModel)
    sbssModel = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(sbssModel)

    sbssModel = Conv2D(128, (3, 3), padding='same', activation='tanh')(sbssModel)
    sbssModel = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(sbssModel)
    sbssModel = Flatten()(sbssModel)
    sbssModel = Dense(units=1024, activation='relu')(sbssModel)
    sbssModel = Dropout(rate=0.4)(sbssModel)
    sbssModel = Dense(3, activation='softmax')(sbssModel)

    model = Model(inputs=inputs, outputs=sbssModel)
    sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=1e-4)
    model.compile(loss=[focal_loss(gamma=2.0)], optimizer=adam, metrics=['accuracy'])
    return model

from keras import backend as K

def focal_loss(gamma=2.):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        return -K.sum(K.pow(1. - pt_1, gamma) * K.log(pt_1))
    return focal_loss_fixed