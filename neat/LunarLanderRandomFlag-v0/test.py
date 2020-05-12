import gym
import neat

from neuroevolution_sandbox.agents.neat_agent import NeatAgent
from neuroevolution_sandbox.env_adapters.gym_env_adapter import GymEnvAdapter
from neat_file_reporter import NeatFileReporter


def main():
    env = gym.make('LunarLanderRandomFlag-v0')

    env_adapter = GymEnvAdapter(env=env, render=True, continuous=False)
    agent = NeatAgent(env_adapter=env_adapter, config_file_path='config.txt')

    agent.load(file_path='trained_model/model')
    while 1:
        agent.play()
main()