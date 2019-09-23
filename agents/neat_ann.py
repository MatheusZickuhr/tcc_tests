import random

import keras
import numpy as np
from keras.constraints import MinMaxNorm
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.optimizers import Adam

import agent_actions
from agents import screen_reader

default_input_shape = (screen_reader.OUTPUT_HEIGHT, screen_reader.OUTPUT_WIDTH, 3)
default_input_shape_resized = (1, screen_reader.OUTPUT_HEIGHT, screen_reader.OUTPUT_WIDTH, 3)

activation_functions = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',
                        'exponential', 'linear']


class Ann:

    def __init__(self, create_model=True):
        self.fitness = 0
        self.dense_layers = list()
        self.model = self.create_model() if create_model else None

    def create_model(self):
        model = Sequential()

        model.add(
            Conv2D(
                128,
                (3, 3),
                input_shape=default_input_shape,
                activation='relu',
                bias_initializer='glorot_uniform',
                kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0),
                bias_constraint=MinMaxNorm(min_value=0.0, max_value=1.0)
            )
        )
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(.2))

        model.add(
            Conv2D(
                128,
                (3, 3),
                input_shape=default_input_shape,
                activation='relu',
                bias_initializer='glorot_uniform',
                kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0),
                bias_constraint=MinMaxNorm(min_value=0.0, max_value=1.0)
            )
        )
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(.2))

        model.add(Flatten())

        model.add(
            Dense(
                128,
                kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0),
                bias_constraint=MinMaxNorm(min_value=0.0, max_value=1.0)
            )
        )

        model.add(
            Dense(
                len(agent_actions.possible_actions), activation='linear', bias_initializer='random_normal',
                kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0),
                bias_constraint=MinMaxNorm(min_value=0.0, max_value=1.0)
            )
        )

        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])

        return model

    def get_next_action(self, img_array):
        img_array = keras.utils.normalize(img_array)
        return np.argmax(
            self.model.predict(
                np.resize(img_array, default_input_shape_resized)
            )
        )

    def reproduce(self, other):
        child_list = (Ann(create_model=False), Ann(create_model=False))

        for child in child_list:
            child.model = Sequential()
            for i in range(len(self.model.layers)):
                parent_layer = self.model.layers[i] if random.random() < 0.5 else other.model.layers[i]
                if type(parent_layer) == Conv2D:
                    is_input_layer = i == 0
                    if is_input_layer:
                        child.model.add(
                            Conv2D(
                                parent_layer.filters,
                                parent_layer.kernel_size,
                                input_shape=default_input_shape,
                                activation=parent_layer.activation,
                                weights=parent_layer.get_weights(),
                                kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0),
                                bias_constraint=MinMaxNorm(min_value=0.0, max_value=1.0)
                            )
                        )
                    else:
                        child.model.add(
                            Conv2D(
                                parent_layer.filters,
                                parent_layer.kernel_size,
                                activation=parent_layer.activation,
                                weights=parent_layer.get_weights(),
                                kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0),
                                bias_constraint=MinMaxNorm(min_value=0.0, max_value=1.0)
                            )
                        )
                elif type(parent_layer) == MaxPooling2D:
                    child.model.add(MaxPooling2D(parent_layer.pool_size))
                elif type(parent_layer) == Dropout:
                    child.model.add(Dropout(parent_layer.rate))
                elif type(parent_layer) == Flatten:
                    child.model.add(Flatten())
                elif type(parent_layer) == Dense:
                    print(parent_layer.units)
                    child.model.add(
                        Dense(
                            parent_layer.units, input_shape=parent_layer.input_shape,
                            weights=parent_layer.get_weights(),
                            kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0),
                            bias_constraint=MinMaxNorm(min_value=0.0, max_value=1.0)
                        )
                    )
            child.model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return child_list

    def mutate(self):
        def mutate_weights(weights):
            for i in range(len(weights)):
                if type(weights[i]) != list:
                    if random.random() < 0.20:
                        weights[i] = random.random()
                else:
                    mutate_weights(weights[i])

        for layer in self.model.layers:
            mutate_weights(layer.get_weights())


class NeatAnn:

    def __init__(self, create_model=True):
        self.fitness = 0
        self.conv_layers_count = random.randint(2, 10)
        self.dense_layers_count = random.randint(2, 10)
        if create_model:
            self.model = self.create_model()

    def create_model(self):
        model = Sequential()

        # input layer
        model.add(
            Conv2D(
                random.randint(10, 256),
                (random.randint(1, 4), random.randint(1, 4)),
                input_shape=default_input_shape,
                bias_initializer='glorot_uniform',
                kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0),
                bias_constraint=MinMaxNorm(min_value=0.0, max_value=1.0),
                activation=random.choice(activation_functions)
            )
        )

        for i in range(self.conv_layers_count):
            model.add(
                Conv2D(
                    random.randint(10, 256),
                    (random.randint(1, 4), random.randint(1, 4)),
                    activation=random.choice(activation_functions),
                    bias_initializer='glorot_uniform',
                    kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0),
                    bias_constraint=MinMaxNorm(min_value=0.0, max_value=1.0)
                )
            )

        model.add(Flatten())

        for i in range(self.dense_layers_count):
            model.add(Dense(random.randint(1, 256), activation=random.choice(activation_functions)))

        model.add(
            Dense(
                len(agent_actions.possible_actions), activation='sigmoid',
                bias_initializer='glorot_uniform',
                kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0),
                bias_constraint=MinMaxNorm(min_value=0.0, max_value=1.0)
            )
        )

        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def reproduce(self, other):
        child1 = NeatAnn(create_model=False)
        child2 = NeatAnn(create_model=False)
        child1.conv_layers_count = self.conv_layers_count
        child1.dense_layers_count = other.dense_layers_count
        child2.conv_layers_count = other.conv_layers_count
        child2.dense_layers_count = self.dense_layers_count

        child1.model = child1.create_model()
        child2.model = child2.create_model()

        return child1, child2

    def mutate(self):
        def mutate_weights(weights):
            for i in range(len(weights)):
                if type(weights[i]) != list:
                    if random.random() < 0.20:
                        weights[i] = random.random()
                else:
                    mutate_weights(weights[i])

        for layer in self.model.layers:
            mutate_weights(layer.get_weights())

    def get_next_action(self, img_array):
        img_array = keras.utils.normalize(img_array)
        predict_value = self.model.predict(
            np.resize(img_array, default_input_shape_resized)
        )[0]
        print predict_value
        return np.argmax(predict_value)
