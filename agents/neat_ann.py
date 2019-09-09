import random

import keras
import numpy as np
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.optimizers import Adam

import agent_actions
from agents import screen_reader

default_input_shape = (screen_reader.OUTPUT_HEIGHT, screen_reader.OUTPUT_WIDTH, 3)
default_input_shape_resized = (1, screen_reader.OUTPUT_HEIGHT, screen_reader.OUTPUT_WIDTH, 3)


class Ann:

    def __init__(self, create_model=True):
        self.dense_layers = list()
        self.model = self.create_model() if create_model else None

    def create_model(self):
        model = Sequential()

        model.add(Conv2D(256, (3, 3), input_shape=default_input_shape, activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(.2))

        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(.2))

        model.add(Flatten())

        self.dense_layers = [
            Dense(random.randint(64, 1024)),
            Dense(random.randint(64, 1024)),
            Dense(random.randint(64, 1024))
        ]

        for dense in self.dense_layers:
            model.add(dense)

        model.add(Dense(len(agent_actions.possible_actions), activation='linear'))

        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])

        return model

    def get_next_action(self, img_array):
        keras.utils.normalize(img_array)
        return np.argmax(
            self.model.predict(
                np.resize(img_array, default_input_shape_resized)
            )
        )

    def reproduce(self, other):
        child_list = (Ann(create_model=False), Ann(create_model=False))

        for child in child_list:

            child.model = Sequential()

            child.model.add(Conv2D(256, (3, 3), input_shape=default_input_shape, activation='relu'))
            child.model.add(MaxPooling2D(2, 2))
            child.model.add(Dropout(.2))

            child.model.add(Conv2D(256, (3, 3), activation='relu'))
            child.model.add(MaxPooling2D(2, 2))
            child.model.add(Dropout(.2))

            child.model.add(Flatten())

            for index in range(len(self.dense_layers)):
                selected_layer = self.dense_layers[index] if random.random() < .5 else other.dense_layers[index]
                child.dense_layers.append(Dense(selected_layer.units))
                child.model.add(child.dense_layers[index])
                child.dense_layers[index].set_weights(selected_layer.get_weights())

            child.model.add(Dense(len(agent_actions.possible_actions), activation='linear'))

            child.model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])

        return child_list

    def mutate(self):
        pass
