import gym
from agents.dqn.dql_agent import DQNAgent
from agents.utils import SRULogger

env = gym.make('LunarLander-v2')

log = SRULogger(file_path='logs/lunar1_other_new_resources_usage_log.txt', log_every_seconds=10 * 60)

agent = DQNAgent(
    env=env,
    input_shape=(10,),
    reward_log_path='logs/lunar1_other_new_reward_log.csv',
)
agent.fit(episodes=100_000, save_model_as='models/lunar1_other_new.model')

log.finish()
