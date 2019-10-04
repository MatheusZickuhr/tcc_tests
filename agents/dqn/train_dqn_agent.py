from agents.dqn.dql_agent import DQNAgent
from ple import PLE
from ple.games import Catcher, MonsterKong

game = MonsterKong()
env = PLE(game, display_screen=False, force_fps=True)
env.init()

agent = DQNAgent(env=env, use_pixels_input=True, input_shape=(50, 50, 3))
agent.fit(episodes=5_000, save_model_as='models\\teste.model')
