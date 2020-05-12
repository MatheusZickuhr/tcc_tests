import gym
import neat

from neuroevolution_sandbox.agents.neat_agent import NeatAgent
from neuroevolution_sandbox.env_adapters.gym_env_adapter import GymEnvAdapter

from log_performance import log_performance
from neat_file_reporter import NeatFileReporter


@log_performance(folder_path='training_data/')
def main():
    env = gym.make('LunarLanderRandomFlag-v0')

    env_adapter = GymEnvAdapter(env=env, render=False, continuous=False)
    agent = NeatAgent(env_adapter=env_adapter, config_file_path='config.txt')

    agent.train(
        number_of_generations=300,
        play_n_times=20,
        max_n_steps=300,
        reward_if_max_step_reached=-200,
        reporters=(
            neat.StdOutReporter(True),
            NeatFileReporter(file_path='training_data/log.csv')
        )
    )
    agent.save(file_path='trained_model/model')
