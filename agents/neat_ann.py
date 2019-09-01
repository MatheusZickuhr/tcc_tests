from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import agent_actions

import os


class Ann:

    def __init__(self):
        self.model = Sequential()

        self.model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(72, 96, 3)))
        self.model.add(Conv2D(32, kernel_size=3, activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(len(agent_actions.possible_actions), activation='softmax'))

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def get_next_action(self, img_array):
        print ' img_array: '
        print img_array
        return np.argmax(
            self.model.predict(
                np.resize(img_array, (1, 72, 96, 3))
            )
        )
