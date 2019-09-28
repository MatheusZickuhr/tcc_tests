import os

os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"

import cv2
import math
import random
import tensorflow as tf
import numpy as np
from keras.models import load_model
from neat_ann import NeatAnn, Ann
from ple.games import FlappyBird
from tqdm import tqdm
from utils import Timer
from ple.games.monsterkong import MonsterKong
from ple import PLE

tf.set_random_seed(1)

game = FlappyBird()
p = PLE(game, display_screen=True, force_fps=True)
p.init()
actions = p.getActionSet()


def get_resized_image(img):
    img = cv2.resize(img, dsize=(10, 10), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32)


class NeatAgent:
    current_element_index = -1
    population = None
    number_of_wins = 0
    number_of_kills = 0

    def start_training(self):
        self.population = self.create_population(size=50)
        generations = 1000

        for i in tqdm(range(generations)):
            for e in self.population:
                p.reset_game()
                done = False
                while not done:
                    state = get_resized_image(p.getScreenRGB())
                    action = e.get_next_action(state)
                    reward = p.act(actions[action])
                    e.fitness += reward
                    done = p.game_over()

            self.crossover_mutate_replace()
        self.save_best_element()

    def play_best(self):

        current = self.load_best_ann()

        done = False
        p.reset_game()
        while not done:
            state = p.getScreenRGB()
            action = current.get_next_action(state)
            p.act(actions[action])

    def create_population(self, size=1):
        population = []
        for i in range(size):
            population.append(Ann(possible_actions=actions))

        return population

    def crossover_mutate_replace(self):
        mutation_chance = 0.1
        reproduction_percentage = .2
        self.population = sorted(self.population, key=lambda x: x.fitness)
        total_fitness = sum([e.fitness for e in self.population])
        number_of_elements = int(len(self.population) * reproduction_percentage)

        if number_of_elements % 2 > 0:
            number_of_elements -= 1

        """
        seleciona uma porcentagem dos elementos, com maior probabildade de selecao
        de elementos com um maior fitness
        """
        selected_elements = []
        for i in range(number_of_elements):

            random_number = random.random()

            for j in range(len(self.population)):

                element = self.population[j]

                if random_number <= sum([e.fitness / total_fitness for e in self.population[:j + 1]]):
                    selected_elements.append(element)

                    break

        children = []
        for i in range(0, len(selected_elements), 2):
            element1 = selected_elements[i]
            element2 = selected_elements[i + 1]

            children.extend(element1.reproduce(element2))

        for child in children:
            if random.random() < mutation_chance:
                child.mutate()

        for i in range(len(children)):
            self.population[i] = children[i]

    def calculate_fitness(self, element=None, time_played=0, time_alive=0):
        fitness = 100 if self.get_player_instance().get_wins() > self.number_of_wins else 1
        fitness *= time_alive
        kills_dif = self.get_player_instance().get_kills() - self.number_of_kills
        fitness *= kills_dif if kills_dif > 0 else 1

        # atualiza os valores para o proxima rodada
        self.number_of_kills = self.get_player_instance().get_kills()
        self.number_of_wins = self.get_player_instance().get_wins()

        element.fitness = fitness


    def has_next_element(self):
        return True if self.current_element_index + 1 < len(self.population) else False

    def save_best_element(self):
        best = sorted(self.population, key=lambda x: x.fitness)[-1]
        best.model.save('neat_agent_best.model')

    def load_best_ann(self):
        ann = NeatAnn(create_model=False)
        ann.model = load_model('neat_agent_best.h5')
        return ann


agent = NeatAgent()
agent.start_training()
