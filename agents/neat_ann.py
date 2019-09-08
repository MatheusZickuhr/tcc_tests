import math
import random

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
import agent_actions

import os

from agents import screen_reader

default_input_shape = (screen_reader.OUTPUT_HEIGHT, screen_reader.OUTPUT_WIDTH, 3)
default_input_shape_resized = (1, screen_reader.OUTPUT_HEIGHT, screen_reader.OUTPUT_WIDTH, 3)


class Ann:

    def __init__(self, is_child=False):
        self.model = Sequential()

        self.conv_layers_count = 0
        self.flatten_layers_count = 0

        if not is_child:

            # layer de entrada
            self.model.add(
                Conv2D(
                    random.randint(32, 512),
                    kernel_size=random.randint(1, 5),
                    activation='relu',
                    input_shape=default_input_shape,
                    kernel_initializer='random_normal',
                    bias_initializer='random_normal'
                )
            )

            # 1 a 4 layers
            self.conv_layers_count = random.randint(1, 4)
            # 1 layer
            self.flatten_layers_count = 1

            for i in range(self.conv_layers_count):
                self.model.add(
                    Conv2D(
                        random.randint(32, 512),
                        kernel_size=random.randint(1, 5),
                        activation='relu',
                        kernel_initializer='random_normal',
                        bias_initializer='random_normal'
                    )
                )

            for i in range(self.flatten_layers_count):
                self.model.add(Flatten())

            # layer de saida
            self.model.add(Dense(len(agent_actions.possible_actions), activation='softmax'))

            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def get_next_action(self, img_array):
        return np.argmax(
            self.model.predict(
                np.resize(img_array, default_input_shape_resized)
            )
        )

    def reproduce(self, other):
        children = (Ann(is_child=True), Ann(is_child=True),)

        for child in children:

            if random.random() < .5:
                child.conv_layers_count = self.conv_layers_count
            else:
                child.conv_layers_count = other.conv_layers_count

            if random.random() < .5:
                child.flatten_layers_count = self.flatten_layers_count
            else:
                child.flatten_layers_count = other.flatten_layers_count

            # layer de entrada
            input_layer = self.model.layers[0] if random.random() < 0.5 else other.model.layers[0]

            input_layer_copy = Conv2D(
                input_layer.filters,
                kernel_size=input_layer.kernel_size,
                activation='relu',
                input_shape=input_layer.input_shape,
                kernel_initializer='random_normal',
                bias_initializer='random_normal'
            )

            input_layer_copy.set_weights(np.copy(input_layer.get_weights))
            child.model.add(input_layer_copy)

            for i in range(child.conv_layers_count):
                possible_layers = []
                possible_layers.extend(
                    list(filter(lambda x: type(x) == keras.layers.convolutional.Conv2D, self.model.layers[1:]))
                )
                possible_layers.extend(
                    list(filter(lambda x: type(x) == keras.layers.convolutional.Conv2D, other.model.layers[1:]))
                )

                new_layer = random.choice(possible_layers)
                possible_layers.remove(new_layer)

                new_layer_copy = Conv2D(
                    new_layer.filters,
                    kernel_size=new_layer.kernel_size,
                    activation='relu',
                    input_shape=new_layer.input_shape,
                    kernel_initializer='random_normal',
                    bias_initializer='random_normal'
                )

                new_layer_copy.set_weights(np.copy(new_layer.get_weights()))

                child.model.add(new_layer_copy)

            for i in range(child.flatten_layers_count):
                possible_layers = []
                possible_layers.extend(
                    list(filter(lambda x: type(x) == keras.layers.core.Flatten, self.model.layers))
                )
                possible_layers.extend(
                    list(filter(lambda x: type(x) == keras.layers.core.Flatten, other.model.layers))
                )

                new_layer = random.choice(possible_layers)
                possible_layers.remove(new_layer)

                new_layer_copy = Flatten()

                new_layer_copy.set_weights(np.copy(new_layer.get_weights()))

                child.model.add(new_layer_copy)

                # layer de saida

            # layer de saida
            child.model.add(Dense(len(agent_actions.possible_actions), activation='softmax'))

        return children

    def mutate(self):
        pass
