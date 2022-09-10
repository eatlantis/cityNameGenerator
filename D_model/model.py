import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


def get_model(num_inputs, pretrained_weights=None):
    input_size = (70, 1)
    # inputs = Input(input_size)
    model = Sequential()
    model.add(LSTM(150))
    model.add(Dense(num_inputs, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=1e-2),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    # model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


