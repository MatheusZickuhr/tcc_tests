import random

import keras
import numpy as np
from keras.constraints import MinMaxNorm
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.optimizers import Adam

activation_functions = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',
                        'exponential', 'linear']


class Ann:

    def __init__(self, create_model=True, n_actions=None, input_shape=None):
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.fitness = 0
        self.dense_layers = list()
        self.model = self.create_model() if create_model else None
        self.was_evaluated = False

    def create_model(self):
        model = Sequential()

        model.add(
            Dense(
                128, input_shape=self.input_shape,
                kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0),
                bias_constraint=MinMaxNorm(min_value=0.0, max_value=1.0)
            )
        )
        model.add(
            Dense(
                128,
                kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0),
                bias_constraint=MinMaxNorm(min_value=0.0, max_value=1.0)
            )
        )

        model.add(
            Dense(
                self.n_actions, activation='linear', bias_initializer='random_normal',
                kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0),
                bias_constraint=MinMaxNorm(min_value=0.0, max_value=1.0)
            )
        )

        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])

        return model

    def get_next_action(self, obs):
        return np.argmax(self.model.predict(obs))

    def reproduce(self, other):
        child1 = Ann(create_model=False, n_actions=self.n_actions, input_shape=self.input_shape)

        child1.model = Sequential()

        child1.model.add(
            Dense(
                self.model.layers[0].units, input_shape=self.input_shape,
                kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0),
                bias_constraint=MinMaxNorm(min_value=0.0, max_value=1.0),
                weights=self.model.layers[0].get_weights()
            )
        )

        child1.model.add(
            Dense(
                other.model.layers[1].units,
                kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0),
                bias_constraint=MinMaxNorm(min_value=0.0, max_value=1.0),
                weights=other.model.layers[1].get_weights()
            )
        )

        child1.model.add(
            Dense(
                self.n_actions, activation='linear', bias_initializer='random_normal',
                kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0),
                bias_constraint=MinMaxNorm(min_value=0.0, max_value=1.0)
            )
        )

        child1.model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])

        child2 = Ann(create_model=False, n_actions=self.n_actions, input_shape=self.input_shape)

        child2.model = Sequential()

        child2.model.add(
            Dense(
                other.model.layers[0].units, input_shape=self.input_shape,
                kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0),
                bias_constraint=MinMaxNorm(min_value=0.0, max_value=1.0),
                weights=other.model.layers[0].get_weights()
            )
        )

        child2.model.add(
            Dense(
                self.model.layers[1].units,
                kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0),
                bias_constraint=MinMaxNorm(min_value=0.0, max_value=1.0),
                weights=self.model.layers[1].get_weights()
            )
        )

        child2.model.add(
            Dense(
                self.n_actions, activation='linear', bias_initializer='random_normal',
                kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0),
                bias_constraint=MinMaxNorm(min_value=0.0, max_value=1.0)
            )
        )

        child2.model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])

        return child1, child2

    def mutate(self):
        def mutate_weights(weights):
            for i in range(len(weights)):
                if type(weights[i]) == np.float32:
                    if random.random() < 0.20:
                        weights[i] = random.random()
                else:
                    mutate_weights(weights[i])

        for layer in self.model.layers:
            new_weights = layer.get_weights()
            mutate_weights(new_weights)
            layer.set_weights(new_weights)
