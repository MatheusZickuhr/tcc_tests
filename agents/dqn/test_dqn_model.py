import gym
import keras
from keras.engine.saving import load_model
import numpy as np
from tqdm import tqdm

env = gym.make('LunarLander-v2')

model = load_model('models/ll1.model')

env.reset()

observation, _, _, _ = env.step(0)
n_games = 10_000
for i in range(n_games):
    done = False
    episode_reward = 0
    while not done:
        observation = keras.utils.normalize(observation)
        action = np.argmax(model.predict(observation))
        observation, reward, done, info = env.step(action)
        episode_reward += reward
        env.render()
    print(f'episode reward: {episode_reward}')
    env.reset()
