import csv
from matplotlib import pyplot as plt
import numpy as np
import json


def cpu_usage_chart(game_name, algorithms):
    cpu_usage_means = []

    for algorithm in algorithms:
        file = open(f'{algorithm}/{game_name}/training_data/Uso de cpu Uso de memória.csv')
        alg_csv = list(csv.reader(file))
        cpu_usage_means.append(np.mean([float(alg_csv[i][0]) for i in range(1, len(alg_csv))]))

    plt.bar(algorithms, cpu_usage_means, width=0.8, bottom=None, align='center', data=None)
    plt.title(f'Uso de cpu médio - {game_name.capitalize()}')
    plt.ylabel('Cpu (%)')
    plt.xlabel('Algoritmos')

    plt.savefig(f'charts/{game_name}_cpu_usage.svg')
    plt.clf()


def memory_usage_chart(game_name, algorithms):
    mem_usage_means = []

    for algorithm in algorithms:
        file = open(f'{algorithm}/{game_name}/training_data/Uso de cpu Uso de memória.csv')
        alg_csv = list(csv.reader(file))
        mem_usage_means.append(np.mean([float(alg_csv[i][1]) for i in range(1, len(alg_csv))]))

    plt.bar(algorithms, mem_usage_means, width=0.8, bottom=None, align='center', data=None)
    plt.title(f'Uso de memória médio - {game_name.capitalize()}')
    plt.ylabel('Memória (MiB)')
    plt.xlabel('Algoritmos')

    plt.savefig(f'charts/{game_name}_memory_usage.svg')
    plt.clf()


def reward_comparicing_chart(game_name, algorithms):
    reward_means = []

    for algorithm in algorithms:
        file = open(f'{algorithm}/{game_name}/trained_model/result.json')
        reward_means.append(np.mean(json.loads(file.read())['rewards']))

    plt.bar(algorithms, reward_means, width=0.8, bottom=None, align='center', data=None)

    plt.savefig(f'charts/{game_name}_reward_comparicing.svg')
    plt.clf()


def time_comparing_chart(game_name, algorithms):
    y = (10.47, 1.26, 1.93)
    x = ('Dql', 'Neat', 'Ne')
    plt.bar(x, y, width=0.8, bottom=None, align='center', data=None)
    plt.title(f'Tempos de treinamento - {game_name.capitalize()}')
    plt.ylabel('Tempo (horas)')
    plt.xlabel('Algoritmos')
    plt.savefig(f'charts/{game_name}_time_comparicing.svg')
    plt.clf()


def show_rewards(game_name, algorithms):
    reward_means = []
    for algorithm in algorithms:
        file = open(f'{algorithm}/{game_name}/trained_model/result.json')
        reward = np.mean(json.loads(file.read())['rewards'])
        reward_means.append(reward)
        print(f'reward mean {algorithm}:', reward)


def show_time_to_train(game_name):
    # dql time to train
    dqn_log = json.loads(open(f'dqn/{game_name}/training_data/log.json').read())
    duration = sum(dqn_log['duration']) / (60 * 60)
    print('time to train DQL:', duration)

    # ne time to train
    ne_log = open(f'ne/{game_name}/training_data/log.csv')
    ne_log_csv = list(csv.reader(ne_log))[1:]
    duration = sum([float(line[3]) for line in ne_log_csv]) / (60 * 60)
    print('time to train NE:', duration)

    # neat time to train
    neat_log = open(f'neat/{game_name}/training_data/log.csv')
    neat_log_csv = list(csv.reader(neat_log))[1:]
    duration = sum([float(line[2]) for line in neat_log_csv]) / (60 * 60)
    print('time to train NEAT:', duration)


def show_cpu_usage(game_name, algorithms):
    for algorithm in algorithms:
        file = open(f'{algorithm}/{game_name}/training_data/Uso de cpu Uso de memória.csv')
        alg_csv = list(csv.reader(file))
        mean = np.mean([float(alg_csv[i][0]) for i in range(1, len(alg_csv))])
        print(f'cpu usage mean {algorithm}:', mean)


def show_memory_usage(game_name, algorithms):
    for algorithm in algorithms:
        file = open(f'{algorithm}/{game_name}/training_data/Uso de cpu Uso de memória.csv')
        alg_csv = list(csv.reader(file))
        mean = np.mean([float(alg_csv[i][1]) for i in range(1, len(alg_csv))])
        print(f'memory usage mean {algorithm}:', mean)


def reward_during_training_chart(game_name):
    sample_size = 100

    dql_reward_samples = []
    dql_log = json.loads(open(f'dqn/{game_name}/training_data/log.json').read())
    episode_reward = dql_log['episode_reward']
    step = len(episode_reward) // sample_size
    for i in range(0, len(episode_reward), step):
        dql_reward_samples.append(episode_reward[i])

    ne_reward_samples = []
    ne_log = open(f'ne/{game_name}/training_data/log.csv')
    ne_log_csv = list(csv.reader(ne_log))[1:]
    generation_reward = [float(line[1]) for line in ne_log_csv]
    step = len(generation_reward) // sample_size
    for i in range(0, len(generation_reward), step):
        ne_reward_samples.append(generation_reward[i])

    neat_reward_samples = []
    neat_log = open(f'neat/{game_name}/training_data/log.csv')
    neat_log_csv = list(csv.reader(neat_log))[1:]
    generation_reward = [float(line[1]) for line in neat_log_csv]
    step = len(generation_reward) // sample_size
    for i in range(0, len(generation_reward), step):
        neat_reward_samples.append(generation_reward[i])

    plt.plot(neat_reward_samples)
    plt.plot(ne_reward_samples)
    plt.plot(dql_reward_samples)
    plt.legend(['NEAT', 'NE', 'DQL'])
    plt.xlabel('Geração/Episódio')
    plt.ylabel('Fitness/Recompensa')
    plt.savefig(f'charts/{game}_reward_during_training.svg')
    plt.clf()


if __name__ == '__main__':
    game = 'flappybird'
    algs = ('dqn', 'neat', 'ne')

    reward_during_training_chart(game)

    # show_rewards(game, algs)
    # show_time_to_train(game)
    # show_cpu_usage(game, algs)
    # show_memory_usage(game, algs)

    # cpu_usage_chart(game, algs)
    # memory_usage_chart(game, algs)
    # reward_comparicing_chart(game, algs)
    # time_comparing_chart(game, algs)
