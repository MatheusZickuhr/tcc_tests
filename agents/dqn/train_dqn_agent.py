from agents.dqn.dql_agent import DQNAgent
from ple import PLE
from ple.games import Catcher, MonsterKong, RaycastMaze

from agents.utils import SRULogger

game = MonsterKong()
env = PLE(game, display_screen=False, force_fps=True)
env.init()

log = SRULogger(file_path='logs\\mk_1_resources_usage_log.txt', log_every_seconds=10 * 60)

agent = DQNAgent(env=env, input_shape=(14,), reward_log_path='logs\\mk_1_reward_log.csv')
agent.fit(episodes=20_000, save_model_as='models\\mk_1.model')

log.finish()
