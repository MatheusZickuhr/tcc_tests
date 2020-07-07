from matplotlib import pyplot as plt

game_name = 'Flappybird'

y = (10.47, 1.26, 1.93)
x = ('Dql', 'Neat', 'Ne')
plt.bar(x, y, width=0.8, bottom=None, align='center', data=None)
plt.title(f'Tempos de treinamento - {game_name.capitalize()}')
plt.ylabel('Tempo (horas)')
plt.xlabel('Algoritmos')
plt.savefig(f'charts/{game_name}_time_comparicing.svg')
