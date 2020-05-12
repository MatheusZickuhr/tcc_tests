import neat

from neuroevolution_sandbox.agents.neat_agent import NeatAgent
from neuroevolution_sandbox.env_adapters.ple_env_adapter import PleEnvAdapter

from log_performance import log_performance
from neat_file_reporter import NeatFileReporter


@log_performance(folder_path='training_data/')
def main():

    env_adapter = PleEnvAdapter(env_name='flappybird', render=False, continuous=False)
    agent = NeatAgent(env_adapter=env_adapter, config_file_path='config.txt')

    agent.train(
        number_of_generations=500,
        reporters=(
            neat.StdOutReporter(True),
            NeatFileReporter(file_path='training_data/log.csv')
        )
    )
    agent.save(file_path='trained_model/model')
