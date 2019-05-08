from keras.layers import Dense, Dropout, Flatten,Input
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD,Adam
from keras.layers import BatchNormalization
from keras.layers import concatenate
from keras.layers.core import Flatten
from keras.models import Model

"""
    怎么样提升模型性能
"""
def sbss_net():
    inputs = Input(shape=(32, 32, 1))

    # path_1
    path_1 = Conv2D(96, (1, 1), strides=(1, 1), padding='same', activation='tanh')(inputs)
    path_1 = BatchNormalization(axis=3)(path_1)
    path_1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='tanh')(path_1)
    path_1 = BatchNormalization(axis=3)(path_1)

    # path_2
    path_2 = Conv2D(16, (1, 1), strides=(1, 1), padding='same', activation='tanh')(inputs)
    path_2 = BatchNormalization(axis=3)(path_2)
    path_2 = Conv2D(32, (5, 5), strides=(1, 1), padding='same', activation='tanh')(path_2)
    path_2 = BatchNormalization(axis=3)(path_2)

    # path_3
    path_3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last')(inputs)
    path_3 = Conv2D(32, (1, 1), strides=(1, 1), activation='tanh')(path_3)
    path_3 = BatchNormalization(axis=3)(path_3)

    # path_4
    path_4 = Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation='tanh')(inputs)
    path_4 = BatchNormalization(axis=3)(path_4)



    sbssModel = concatenate([path_1, path_2, path_3, path_4], axis=3)
    sbssModel = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(sbssModel)
    sbssModel = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(sbssModel)
    sbssModel = Flatten()(sbssModel)
    sbssModel = Dense(units=1024, activation='relu')(sbssModel)
    sbssModel = Dropout(rate=0.4)(sbssModel)
    sbssModel = Dense(3, activation='softmax')(sbssModel)

    model = Model(inputs=inputs, outputs=sbssModel)
    sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

