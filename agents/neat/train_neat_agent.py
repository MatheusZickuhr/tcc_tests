from agents.neat.neat_agent import NeatAgent
import gym

from agents.utils import SRULogger

env = gym.make('LunarLander-v2')

log = SRULogger(file_path='logs/lunar1_resources_usage_log.txt', log_every_seconds=10 * 60)


agent = NeatAgent(env=env, population_size=100, input_shape=(8,), reward_log_path='logs/lunar1_neat_reward_log')
agent.fit(generations=150, save_as='models/test.model')

log.finish()
