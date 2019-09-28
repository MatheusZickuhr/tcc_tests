import os

from keras.engine.saving import load_model

os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"

import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from collections import deque

from ple.games import Pong
from tqdm import tqdm
import os
from ple.games.flappybird import FlappyBird
from ple import PLE
import random
import tensorflow as tf


DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 64
UPDATE_TARGET_EVERY = 5
MODEL_NAME = '2x256'
MIN_REWARD = -200
MEMORY_FRACTION = 0.20

EPISODES = 20_000

epsilon = 1
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 50
SHOW_PREVIEW = False

random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

game = FlappyBird()
p = PLE(game, display_screen=True, force_fps=True)
p.init()
actions = p.getActionSet()

default_input_shape = (10, 10, 3)

should_load_model = 'dql_fb.model'


def get_resized_image(img):
    img = cv2.resize(img, dsize=default_input_shape[:2], interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32)


class DQNAgent:

    def __init__(self):
        self.model = load_model(should_load_model) if should_load_model else self.create_model()

        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.target_update_counter = 0

        self.epsilon = 1

    def create_model(self):
        model = Sequential()

        model.add(Conv2D(256, (3, 3), input_shape=default_input_shape,
                         use_bias=True))
        # model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3), use_bias=True))
        # model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())

        model.add(Dense(64, use_bias=True))
        # model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Dense(len(actions), use_bias=True))
        # model.add(BatchNormalization())
        model.add(Activation('linear'))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, terminal_state):

        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch]) / 255
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch]) / 255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(
            np.array(X) / 255, np.array(y),
            batch_size=MINIBATCH_SIZE,
            verbose=0,
            shuffle=False,
            callbacks=None
        )

        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]


agent = DQNAgent()

for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    episode_reward = 0
    step = 1

    p.reset_game()
    current_state = get_resized_image(p.getScreenRGB())

    done = False
    while not done:
        if np.random.random() > epsilon:
            action_index = np.argmax(agent.get_qs(current_state))
            action = actions[action_index]
        else:
            action_index = random.randint(0, len(actions) - 1)
            action = actions[action_index]

        reward = p.act(action)
        new_state = get_resized_image(p.getScreenRGB())
        done = p.game_over()

        episode_reward += reward

        agent.update_replay_memory((current_state, action_index, reward, new_state, done))

        agent.train(done)

        current_state = new_state
        step += 1

    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

agent.model.save('dql_fb.model')
