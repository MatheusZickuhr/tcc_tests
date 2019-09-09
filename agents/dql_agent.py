import random
import time
from collections import deque

import numpy as np
from keras import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam

from agents import screen_reader, agent_actions

default_input_shape = (screen_reader.OUTPUT_HEIGHT, screen_reader.OUTPUT_WIDTH, 3)
default_input_shape_resized = (1, screen_reader.OUTPUT_HEIGHT, screen_reader.OUTPUT_WIDTH, 3)

REPLAY_MEMORY_SIZE = 50000

MODEL_NAME = '256x2'

MIN_REPLAY_MEMORY_SZIE = 100
MINIBATCH_SIZE = 64

DISCONT = 0.99


class DQLAgent:

    def __init__(self):
        #  treinado
        self.model = self.create_model()

        # usado para prever (.predict)
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorbord = ModifiedTensorBoard(log_dir='logs/{}-{}'.format(MODEL_NAME, int(time.time())))

        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(256, (3, 3), input_shape=default_input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(.2))

        model.add(Flatten())
        model.add(Dense(64))

        self.model.add(Dense(len(agent_actions.possible_actions), activation='linear'))

        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])

        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state, step):
        return self.model_predict(np.array(state).reshape(-1, *state.shape) / 255)[0]

    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SZIE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch]) / 255
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch]) / 255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        Y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCONT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            Y.append(current_qs)


class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super(**kwargs)
        self.step = 1

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)
