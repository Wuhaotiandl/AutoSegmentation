from keras.layers import Dense, Dropout, Flatten,Input
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD,Adam
from keras.layers import BatchNormalization
from keras.layers import concatenate
from keras.layers.core import Flatten
from keras.models import Model

def sbss_net():
    inputs = Input(shape=(32, 32, 1))
    sbssModel = Conv2D(64, (5, 5), padding='same', activation='tanh')(inputs)
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
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

