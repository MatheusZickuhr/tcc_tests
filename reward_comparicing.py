from matplotlib import pyplot as plt
import json

game_name = 'flappybird'

dql_file = open(f'dqn/{game_name}/trained_model/result.json')
neat_file = open(f'neat/{game_name}/trained_model/result.json')
ne_file = open(f'ne/{game_name}/trained_model/result.json')

dql_rewards = json.loads(dql_file.read())['rewards']
neat_rewards = json.loads(neat_file.read())['rewards']
ne_rewards = json.loads(ne_file.read())['rewards']

plt.plot((1, 2, 3), (4, 5, 6))
plt.savefig(f'charts/{game_name}_reward_comparicing.svg')
