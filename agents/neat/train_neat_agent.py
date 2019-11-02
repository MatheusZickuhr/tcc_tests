from agents.neat.neat_agent import NeatAgent
import gym

from agents.utils import SRULogger

env = gym.make('LunarLander-v2')

log = SRULogger(file_path='logs/nrf1__neat_resources_usage_log.txt', log_every_seconds=10 * 60)


agent = NeatAgent(env=env, population_size=500, input_shape=(10,), reward_log_path='logs/rf1_neat_reward_log')
agent.fit(generations=200, save_as='models/nrf1_neat.model')

log.finish()
