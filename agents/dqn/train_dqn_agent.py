from agents.dqn.dql_agent import DQNAgent
from ple import PLE
from ple.games import Catcher

game = Catcher()
env = PLE(game, display_screen=False, force_fps=True)
env.init()

agent = DQNAgent(env=env, input_shape=(10, 10, 3))
agent.fit(episodes=30_000, save_model_as='models\\catcher_model.model')
