import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from collections import deque
import random
from tqdm import tqdm
import os

from agents import agent_actions, screen_reader
from agents.base_agent import Agent
from agents.screen_reader import get_next_frame


default_input_shape = (screen_reader.OUTPUT_HEIGHT, screen_reader.OUTPUT_WIDTH, 3)
default_input_shape_resized = (1, screen_reader.OUTPUT_HEIGHT, screen_reader.OUTPUT_WIDTH, 3)

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = '2x256'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 20000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False


# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
# tf.set_random_seed(1)

# Memory fraction, used mostly when trai8ning multiple agents
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
# backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


# Agent class
class DQNAgent(Agent, object):

    def initialize(self):
        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

        # Exploration settings
        self.epsilon = 1

    def create_model(self):
        model = Sequential()

        model.add(Conv2D(256, (3, 3),
                         input_shape=default_input_shape))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))

        model.add(Dense(len(agent_actions.possible_actions), activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch]) / 255
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch]) / 255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X) / 255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]

    def get_reward(self):
        return -100 if self.is_player_dead() else 100

    def do_action(self, action):
        super(DQNAgent, self).do_action(action)
        new_state = get_next_frame()
        reward = self.get_reward()
        done = self.is_player_dead()
        return new_state, reward, done

    def start(self):
        self.wait_game_to_start()

        # Iterate over episodes
        for i in tqdm(range(self.get_total_games_count())):

            self.wait_agent_can_play()

            # Restarting episode - reset episode reward and step number
            episode_reward = 0
            step = 1

            # Reset environment and get initial state
            current_state = get_next_frame()

            # Reset flag and start iterating until episode ends
            done = False
            while not done:

                # This part stays mostly the same, the change is to query a model for Q values
                if np.random.random() > self.epsilon:
                    # Get action from Q table
                    action = np.argmax(self.get_qs(current_state))
                else:
                    # Get random action
                    action = random.choice(agent_actions.possible_actions)

                new_state, reward, done = self.do_action(action)

                # Transform new continous state to new discrete state and count reward
                episode_reward += reward

                # Every step we update replay memory and train main network
                self.update_replay_memory((current_state, action, reward, new_state, done))
                self.train(done, step)

                current_state = new_state
                step += 1

            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])

            ep_rewards.append(episode_reward)

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:
                self.model.save('my_model.model')

            # Decay epsilon
            if self.epsilon > MIN_EPSILON:
                self.epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, self.epsilon)


# agent = DQNAgent()
#
# # Iterate over episodes
# for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
#
#     # Update tensorboard step every episode
#     agent.tensorboard.step = episode
#
#     # Restarting episode - reset episode reward and step number
#     episode_reward = 0
#     step = 1
#
#     # Reset environment and get initial state
#     current_state = env.reset()
#
#     # Reset flag and start iterating until episode ends
#     done = False
#     while not done:
#
#         # This part stays mostly the same, the change is to query a model for Q values
#         if np.random.random() > epsilon:
#             # Get action from Q table
#             action = np.argmax(agent.get_qs(current_state))
#         else:
#             # Get random action
#             action = np.random.randint(0, env.ACTION_SPACE_SIZE)
#
#         new_state, reward, done = env.step(action)
#
#         # Transform new continous state to new discrete state and count reward
#         episode_reward += reward
#
#         if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
#             env.render()
#
#         # Every step we update replay memory and train main network
#         agent.update_replay_memory((current_state, action, reward, new_state, done))
#         agent.train(done, step)
#
#         current_state = new_state
#         step += 1
#
#     # Append episode reward to a list and log stats (every given number of episodes)
#     ep_rewards.append(episode_reward)
#     if not episode % AGGREGATE_STATS_EVERY or episode == 1:
#         average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
#         min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
#         max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
#         agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
#                                        epsilon=epsilon)
#
#         # Save model, but only when min reward is greater or equal a set value
#         if min_reward >= MIN_REWARD:
#             agent.model.save(
#                 f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(
#                     time.time())}.model')
#
#     # Decay epsilon
#     if epsilon > MIN_EPSILON:
#         epsilon *= EPSILON_DECAY
#         epsilon = max(MIN_EPSILON, epsilon)
