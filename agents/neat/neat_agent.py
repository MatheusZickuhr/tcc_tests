import copy
import random

import cv2
import keras
import numpy as np
from tqdm import tqdm

from agents.neat.neat_ann import Ann


class NeatAgent:

    def __init__(self, env=None, input_shape=None, population_size=100):
        self.env = env
        self.input_shape = input_shape
        self.population = self.create_population(size=population_size)

    def fit(self, generations=1000, save_as=None):
        for generation in tqdm(range(generations)):
            for element in self.population:
                self.env.reset()
                done = False
                observation, _, _, _ = self.env.step(self.env.action_space.sample())
                while not done:
                    observation = keras.utils.normalize(observation)
                    action = element.get_next_action(observation)
                    observation, reward, done, info = self.env.step(action)
                    element.fitness += reward if reward >= 0 else 0

            self.crossover_mutate_replace()
        self.save_best_element(save_as)

    def encode(self, ann):
        return ann.model.get_weights()

    def decode(self, encoded_ann):
        ann = Ann(input_shape=self.input_shape, n_actions=len(self.env.getActionSet()))
        ann.model.set_weights(encoded_ann)
        return ann

    def create_population(self, size=1):
        population = []
        for i in tqdm(range(size), unit='population element'):
            population.append(Ann(input_shape=self.input_shape, n_actions=self.env.action_space.n))

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
        mutation_chance = 0.1
        reproduction_percentage = .2
        self.population = sorted(self.population, key=lambda x: x.fitness)
        number_of_elements = int(len(self.population) * reproduction_percentage)

        if number_of_elements % 2 > 0:
            number_of_elements -= 1

        """
        seleciona uma porcentagem dos elementos, com maior probabildade de selecao
        de elementos com um maior fitness
        """
        selected_elements = []
        temp_polulation = copy.copy(self.population)
        total_fitness = sum([e.fitness for e in temp_polulation])
        for i in range(number_of_elements):
            if total_fitness == 0:
                random_element = random.choice(temp_polulation)
                selected_elements.append(random_element)
                temp_polulation.remove(random_element)
                total_fitness -= random_element.fitness
            else:
                random_number = random.random()

                for j in range(len(temp_polulation)):

                    element = temp_polulation[j]

                    if random_number <= sum([e.fitness / total_fitness for e in temp_polulation[:j + 1]]):
                        selected_elements.append(element)
                        temp_polulation.remove(element)
                        total_fitness -= element.fitness
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

    def save_best_element(self, save_as):
        best = sorted(self.population, key=lambda x: x.fitness)[-1]
        best.model.save(save_as)
