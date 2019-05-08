from keras.layers import Dense, Dropout, Flatten,Input
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD,Adam
from keras.layers import BatchNormalization
from keras.layers import concatenate
from keras.layers.core import Flatten
from keras.models import Model

def sbss_net():
    inputs = Input(shape=(32, 32, 1))

    #path_1
    sbssModel_1 = Conv2D(64, (5, 5), padding='same', activation='tanh')(inputs)
    sbssModel_1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(sbssModel_1)
    sbssModel_1 = Conv2D(128, (3, 3), padding='same', activation='tanh')(sbssModel_1)
    sbssModel_1 = BatchNormalization(axis=3)(sbssModel_1)
    sbssModel_1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(sbssModel_1)

    #path_2
    sbssModel_2 = Conv2D(64, (1, 1), padding='same', activation='tanh')(inputs)
    sbssModel_2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(sbssModel_2)
    sbssModel_2 = Conv2D(128, (3, 3), padding='same', activation='tanh')(sbssModel_2)
    sbssModel_2 = BatchNormalization(axis=3)(sbssModel_2)
    sbssModel_2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(sbssModel_2)

    sbssModel = concatenate([sbssModel_1, sbssModel_2], axis=3)
    sbssModel = Flatten()(sbssModel)
    sbssModel = Dense(units=1024, activation='relu')(sbssModel)
    sbssModel = Dropout(rate=0.4)(sbssModel)
    sbssModel = Dense(3, activation='softmax')(sbssModel)

    model = Model(inputs=inputs, outputs=sbssModel)
    sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

