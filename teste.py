import random

import keras
import numpy as np
from keras.constraints import MinMaxNorm
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from agents import screen_reader

default_input_shape = (screen_reader.OUTPUT_HEIGHT, screen_reader.OUTPUT_WIDTH, 3)
default_input_shape_resized = (1, screen_reader.OUTPUT_HEIGHT, screen_reader.OUTPUT_WIDTH, 3)

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
        5, activation='linear', bias_initializer='random_normal',
        kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0),
        bias_constraint=MinMaxNorm(min_value=0.0, max_value=1.0)
    )
)

model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])

other_model = Sequential()

other_model.add(
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
other_model.add(MaxPooling2D(2, 2))
other_model.add(Dropout(.2))

other_model.add(
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
other_model.add(MaxPooling2D(2, 2))
other_model.add(Dropout(.2))

other_model.add(Flatten())

parent_layer = model.layers[3]

for layer in model.layers:
    print(type(layer))

other_model.add(
    Dense(
        parent_layer.units, input_shape=parent_layer.input_shape,
        weights=parent_layer.get_weights(),
        kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0),
        bias_constraint=MinMaxNorm(min_value=0.0, max_value=1.0)
    )
)
