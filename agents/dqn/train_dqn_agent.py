from agents.dqn.dql_agent import DQNAgent
from ple import PLE
from ple.games import FlappyBird

game = FlappyBird()
env = PLE(game, display_screen=True, force_fps=True)
env.init()

agent = DQNAgent(input_shape=(10, 10, 3), env=env, model_path='models\\dql_fb.model')
agent.fit(episodes=20_0000)
