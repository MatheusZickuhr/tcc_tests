import gym
import keras
from keras.engine.saving import load_model
import numpy as np

env = gym.make('LunarLander-v2')

model = load_model('models/rf1.model')

env.reset()

observation, _, _, _ = env.step(0)
total_reward = 0
while True:
    observation = keras.utils.normalize(observation)
    action = np.argmax(model.predict(observation))
    observation, reward, done, info = env.step(action)
    env.render()
    total_reward += reward
    if done:
        print(total_reward)
        total_reward = 0
        env.reset()
