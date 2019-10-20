from agents.neat.neat_agent import NeatAgent
import gym

env = gym.make('LunarLander-v2')

agent = NeatAgent(env=env, population_size=100, input_shape=(8,))
agent.fit(generations=100, save_as='models/test.model')
