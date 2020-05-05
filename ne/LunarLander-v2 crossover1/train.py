import gym

from python_ne.core.ga.console_logger import ConsoleLogger
from python_ne.core.ga.crossover_strategies import Crossover1
from python_ne.core.ga.csv_logger import CsvLogger
from python_ne.core.ga.matplotlib_logger import MatplotlibLogger
from python_ne.core.ga.mutation_strategies import Mutation1
from python_ne.core.model_adapters.default_model_adapter import DefaultModelAdapter
from python_ne.extra.env_adapters.gym_env_adapter import GymEnvAdapter
from python_ne.extra.ne_agent import NeAgent
from log_performance import log_performance


@log_performance(folder_path='training_data')
def main():
    env = gym.make('LunarLander-v2')

    env_adapter = GymEnvAdapter(env=env, render=False, continuous=False)

    agent = NeAgent(
        env_adapter=env_adapter,
        model_adapter=DefaultModelAdapter,
    )

    nn_config = (
        (env_adapter.get_input_shape(), 16, 'tanh'),
        (env_adapter.get_n_actions(), 'tanh')
    )

    csv_logger = CsvLogger()
    matplotlib_logger = MatplotlibLogger()
    console_logger = ConsoleLogger()

    agent.train(
        number_of_generations=1000,
        population_size=500,
        selection_percentage=0.9,
        mutation_chance=0.01,
        fitness_threshold=250,
        neural_network_config=nn_config,
        crossover_strategy=Crossover1(),
        mutation_strategy=Mutation1(),
        play_n_times=5,
        max_n_steps=300,
        reward_if_max_step_reached=-200,
        loggers=(console_logger, csv_logger, matplotlib_logger)
    )
    csv_logger.save('training_data/log.csv')
    matplotlib_logger.save_fitness_chart(
        'training_data/fitness.png',
        fitness_label='Fitness',
        generation_label='Geração',
        chart_title='Geração - Fitness'
    )
    matplotlib_logger.save_std_chart(
        'training_data/std.png',
        std_label='Desvio padrão',
        generation_label='Geração',
        chart_title='Geração - Desvio padrão'
    )
    matplotlib_logger.save_time_chart(
        'training_data/time.png',
        time_label='Tempo (s)',
        generation_label='Geração',
        chart_title='Geração - Tempo'
    )
    agent.save('trained_model/model.json')
