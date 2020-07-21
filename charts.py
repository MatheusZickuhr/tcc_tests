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


if __name__ == '__main__':
    game = 'LunarLander-v2'
    algs = ('dqn', 'neat', 'ne')

    cpu_usage_chart(game, algs)
    memory_usage_chart(game, algs)
    reward_comparicing_chart(game, algs)
    time_comparing_chart(game, algs)
