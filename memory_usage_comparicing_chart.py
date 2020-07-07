import csv
from matplotlib import pyplot as plt
import numpy as np

game_name = 'flappybird'

ne_file = open(f'ne/{game_name}/training_data/Uso de cpu Uso de memória.csv')
neat_file = open(f'neat/{game_name}/training_data/Uso de cpu Uso de memória.csv')
dql_file = open(f'dqn/{game_name}/training_data/Uso de cpu Uso de memória.csv')

ne_csv = list(csv.reader(ne_file))
neat_csv = list(csv.reader(neat_file))
dql_csv = list(csv.reader(dql_file))

ne_mem_usage = np.mean([float(ne_csv[i][1]) for i in range(1, len(ne_csv))])
neat_mem_usage = np.mean([float(neat_csv[i][1]) for i in range(1, len(neat_csv))])
dql_mem_usage = np.mean([float(dql_csv[i][1]) for i in range(1, len(dql_csv))])

x = ('Dql', 'Neat', 'Ne')
y = (dql_mem_usage, neat_mem_usage, ne_mem_usage)

plt.bar(x, y, width=0.8, bottom=None, align='center', data=None)
plt.title(f'Uso de memória médio - {game_name.capitalize()}')
plt.ylabel('Memória (MiB)')
plt.xlabel('Algoritmos')

plt.savefig(f'charts/{game_name}_memory_usage.svg')
