import gym
import keras
from keras.engine.saving import load_model
import numpy as np

env = gym.make('LunarLander-v2')

model = load_model('models/lunar1.model')

env.reset()

observation, _, _, _ = env.step(0)
while True:
    observation = keras.utils.normalize(observation)
    action = np.argmax(model.predict(observation))
    observation, reward, done, info = env.step(action)
    env.render()
    if done:
        env.reset()
