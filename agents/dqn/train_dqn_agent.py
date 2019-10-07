from agents.dqn.dql_agent import DQNAgent
from ple import PLE
from ple.games import Catcher, MonsterKong, RaycastMaze

from agents.utils import SRULogger

game = MonsterKong()
env = PLE(game, display_screen=True, force_fps=True)
env.init()

log = SRULogger(file_path='logs\\log.txt', log_every_seconds=60)

agent = DQNAgent(env=env, use_pixels_input=True, input_shape=(30, 30, 3), reward_log_path='logs\\teste_reward_log.csv')
agent.fit(episodes=3, save_model_as='models\\teste.model')

log.finish()
