import keras
from keras.engine.saving import load_model
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from collections import deque
from tqdm.auto import tqdm
import random
import tensorflow as tf

random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)


class DQNAgent:

    def __init__(
            self,
            replay_memory_size=50_000,
            input_shape=None,
            discount=0.99,
            update_target_every=5,
            env=None,
            model_path=None,
            reward_log_path=None,
            use_pixels_input=False
    ):
        self.reward_log_path = reward_log_path
        self.use_pixels_input = use_pixels_input
        self.epsilon = 1
        self.min_epsilon = 0.001
        self.save_model_every = 500
        self.input_shape = input_shape
        self.n_actions = len(env.getActionSet())
        self.min_replay_memory_size = 1000
        self.minibatch_size = 64
        self.discount = discount
        self.update_target_every = update_target_every
        self.env = env
        if self.use_pixels_input:
            self.model = self.create_conv_model() if model_path is None else load_model(model_path)
            self.target_model = self.create_conv_model() if model_path is None else load_model(model_path)
            self.input_shape = env.getScreenRGB().shape if self.input_shape is None else self.input_shape
        else:
            self.model = self.create_dense_model() if model_path is None else load_model(model_path)
            self.target_model = self.create_dense_model() if model_path is None else load_model(model_path)
            self.input_shape = len(env.getGameState().keys()) if self.input_shape is None else self.input_shape
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=replay_memory_size)
        self.target_update_counter = 0

    def create_conv_model(self):
        model = Sequential()

        model.add(Conv2D(256, (3, 3), input_shape=self.input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3), use_bias=True))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())

        model.add(Dense(64, use_bias=True))
        model.add(Activation('relu'))

        model.add(Dense(self.n_actions, use_bias=True))
        model.add(Activation('linear'))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, terminal_state):

        if len(self.replay_memory) < self.min_replay_memory_size:
            return

        minibatch = random.sample(self.replay_memory, self.minibatch_size)

        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.discount * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(
            np.array(X), np.array(y),
            batch_size=self.minibatch_size,
            verbose=0,
            shuffle=False,
            callbacks=None
        )

        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > self.update_target_every:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array([state]))

    def normalize_game_state(self, state):
        normalized_state = []
        for key in sorted(state.keys()):
            normalized_state.append(state[key])
        return keras.utils.normalize(normalized_state).reshape(self.input_shape)

    def create_dense_model(self):
        model = Sequential()
        model.add(Dense(input_shape=self.input_shape, units=128))
        model.add(Dense(units=128))
        model.add(Dense(self.n_actions, use_bias=True))
        model.add(Activation('linear'))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def resize_and_normalize_img(self, img):
        if self.input_shape:
            img = cv2.resize(img, self.input_shape[:2], interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return keras.utils.normalize(img.astype(np.float32))

    def get_current_env_state(self):
        if self.use_pixels_input:
            return self.resize_and_normalize_img(self.env.getScreenRGB())
        else:
            return self.normalize_game_state(self.env.getGameState())

    def recalc_epsilon(self, epsilon_decay, resume_from_episode):
        for i in range(resume_from_episode):
            if self.epsilon > self.min_epsilon:
                self.epsilon *= epsilon_decay
                self.epsilon = max(self.min_epsilon, self.epsilon)

    def log_reward(self, path, episode=0, reward=0):
        with open(path, 'a') as file:
            file.write(f'{episode},{reward}\n')

    def fit(self, episodes=20_000, save_model_as='model_name.model', resume_from_episode=0):
        epsilon_decay = 1 - (5 / episodes)
        self.recalc_epsilon(epsilon_decay, resume_from_episode)
        for episode in tqdm(range(1, episodes + 1), ascii=True, unit='episodes'):
            if episode < resume_from_episode:
                continue

            episode_reward = 0
            self.env.reset_game()
            current_state = self.get_current_env_state()
            done = False
            while not done:
                action_index = np.argmax(self.get_qs(current_state)) if np.random.random() > self.epsilon \
                    else random.randint(0, self.n_actions - 1)
                action = self.env.getActionSet()[action_index]
                reward = self.env.act(action)
                new_state = self.get_current_env_state()
                done = self.env.game_over()
                self.update_replay_memory((current_state, action_index, reward, new_state, done))
                self.train(done)
                current_state = new_state
                episode_reward += reward

            if self.epsilon > self.min_epsilon:
                self.epsilon *= epsilon_decay
                self.epsilon = max(self.min_epsilon, self.epsilon)

            if episode % self.save_model_every == 0 or episode == 1 or episode == episodes:
                self.model.save(save_model_as)
                if self.reward_log_path:
                    self.log_reward(path=self.reward_log_path, episode=episode, reward=episode_reward)

