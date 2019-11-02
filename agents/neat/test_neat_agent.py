import gym
import keras
from keras.engine.saving import load_model
import numpy as np

env = gym.make('LunarLander-v2')

model = load_model('models/nrf1_neat.model')

env.reset()

observation, _, _, _ = env.step(0)
total_reward = 0
n_games = 1000
for _ in range(n_games):
    ep_reward = 0
    done = False
    while not done:
        observation = keras.utils.normalize(observation)
        action = np.argmax(model.predict(observation))
        observation, reward, done, info = env.step(action)
        # env.render()
        ep_reward += reward
        if done:
            print(ep_reward)
            total_reward += ep_reward
            env.reset()

print(f'avg reward {total_reward / n_games + 1}')
