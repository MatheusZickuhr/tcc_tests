import gym
import keras
from keras.engine.saving import load_model
import numpy as np
from tqdm import tqdm

env = gym.make('LunarLander-v2')

model = load_model('models/lunar1_other_new.model')

env.reset()

observation, _, _, _ = env.step(0)
n_games = 10_000
for i in tqdm(range(n_games)):
    done = False
    while not done:
        observation = keras.utils.normalize(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        # env.render()
    env.reset()
