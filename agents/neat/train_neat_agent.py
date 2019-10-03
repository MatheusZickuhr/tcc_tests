from agents.neat.neat_agent import NeatAgent
from ple import PLE
from ple.games import Catcher

game = Catcher()
env = PLE(game, display_screen=True, force_fps=True)
env.init()

agent = NeatAgent(env=env, population_size=100, input_shape=(10, 10, 3))
agent.fit(generations=10, save_as='models/test.model')
