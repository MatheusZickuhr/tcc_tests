import keras
import cv2
import random
import tensorflow as tf
import numpy as np
from keras.models import load_model
from agents.neat.neat_ann import NeatAnn, Ann
from tqdm import tqdm

tf.set_random_seed(1)


class NeatAgent:

    def __init__(self, env=None, input_shape=None, population_size=100):
        self.population = self.create_population(size=population_size)
        self.env = env
        self.input_shape = input_shape

    def fit(self, generations=1000, save_as=None):
        for generation in tqdm(range(generations)):
            for element in self.population:
                temp_element = self.decode(element)
                done = False
                while not done:
                    state = self.resize_and_normalize_img(self.env.getScreenRGB())
                    action = temp_element.get_next_action(state)
                    reward = self.env.act(self.env.getActionSet()[action])
                    temp_element.fitness += reward
                    done = self.env.game_over()

        self.crossover_mutate_replace()
        self.save_best_element(save_as)

    def encode(self, ann):
        return ann.model.get_weights()

    def decode(self, encoded_ann):
        ann = Ann()
        ann.model.set_weights(encoded_ann)
        return ann

    def create_population(self, size=1):
        population = []
        for i in range(size):
            population.append(self.encode(Ann(n_actions=len(self.env.getActionSet()))))

        return population

    def resize_and_normalize_img(self, img):
        if self.input_shape:
            img = cv2.resize(img, dsize=self.input_shape[:2], interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return keras.utils.normalize(img.astype(np.float32))

    def decode_all_population(self):
        self.population = [self.decode(e) for e in self.population]

    def encode_all_population(self):
        self.population = [self.encode(e) for e in self.population]

    def crossover_mutate_replace(self):
        self.decode_all_population()

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

        self.encode_all_population()

    def calculate_fitness(self, element=None, time_played=0, time_alive=0):
        fitness = 100 if self.get_player_instance().get_wins() > self.number_of_wins else 1
        fitness *= time_alive
        kills_dif = self.get_player_instance().get_kills() - self.number_of_kills
        fitness *= kills_dif if kills_dif > 0 else 1

        # atualiza os valores para o proxima rodada
        self.number_of_kills = self.get_player_instance().get_kills()
        self.number_of_wins = self.get_player_instance().get_wins()

        element.fitness = fitness

    def save_best_element(self, save_as):
        best = sorted(self.population, key=lambda x: x.fitness)[-1]
        best.model.save(save_as)
