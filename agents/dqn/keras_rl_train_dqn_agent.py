import numpy as np
import gym
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from agents.utils import SRULogger

env = gym.make('LunarLander-v2')
np.random.seed(123)
env.seed(123)

model = Sequential()
model.add(Flatten(input_shape=(1,) + (10,)))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(env.action_space.n))
model.add(Activation('linear'))

log = SRULogger(file_path='logs/256_128_resources_usage_log.txt', log_every_seconds=10 * 60)

dqn = DQNAgent(
    model=model,
    nb_actions=env.action_space.n,
    memory=SequentialMemory(limit=65000, window_length=1),
    nb_steps_warmup=1000,
    target_model_update=1e-2,
    policy=EpsGreedyQPolicy()
)
dqn.compile(Adam(lr=0.001), metrics=['mae'])
# dqn.load_weights('models/rf1.model')
dqn.fit(env, nb_steps=1_100_000, visualize=False, verbose=2)

log.finish()

dqn.save_weights('models/256_128.model', overwrite=True)
dqn.test(env, nb_episodes=1000, visualize=True)
