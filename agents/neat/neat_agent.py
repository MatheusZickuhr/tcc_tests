import copy
import random
import keras
from tqdm import tqdm
from agents.neat.neat_ann import NeatAnn
import gc


class NeatAgent:

    def __init__(self, env=None, input_shape=None, population_size=100, reward_log_path=''):
        self.env = env
        self.input_shape = input_shape
        self.population = self.create_population(size=population_size)
        self.log_reward_every = 1
        self.reward_log_path = reward_log_path
        self.is_population_encoded = False

    def fit(self, generations=1000, save_as=None):
        for generation in tqdm(range(generations)):
            if self.is_population_encoded:
                self.decode()

            self.reset_population_fitness()

            for element in self.population:
                self.env.reset()
                done = False
                observation, _, _, _ = self.env.step(self.env.action_space.sample())
                while not done:
                    observation = keras.utils.normalize(observation)
                    action = element.get_next_action(observation)
                    observation, reward, done, info = self.env.step(action)
                    element.fitness += reward if reward > 0 else 0
                element.was_evaluated = True
            # self.normalize_polulation_fitness()
            self.crossover_mutate_replace()

            if generation == 0 or generation % self.log_reward_every == 0 or generation == generations - 1:
                self.log_reward(self.reward_log_path, generation=generation, reward=self.population[-1].fitness)

            self.encode()
            keras.backend.clear_session()
            gc.collect()

        self.save_best_element(save_as)

    def reset_population_fitness(self):
        for e in self.population:
            e.fitness = 0

    def encode(self):
        self.is_population_encoded = True
        encoded_population = []
        for e in self.population:
            encoded_population.append(
                {
                    'weights': e.model.get_weights(),
                    'fitness': e.fitness,
                    'was_evaluated': e.was_evaluated,
                    'topology': e.topology
                }
            )
            del e.model
        self.population = encoded_population

    def decode(self):
        self.is_population_encoded = False
        decoded_population = []
        for e in self.population:
            ann = NeatAnn(input_shape=self.input_shape, n_actions=self.env.action_space.n, topology=e['topology'])
            ann.model.set_weights(e['weights'])
            ann.fitness = e['fitness']
            ann.was_evaluated = e['was_evaluated']
            decoded_population.append(ann)
        self.population = decoded_population

    def normalize_polulation_fitness(self):
        def normalize(arr):
            new_arr = []
            for e in arr:
                new_arr.append((e - min(arr)) / (max(arr) - min(arr)))
            return new_arr

        fitness_list = [e.fitness for e in self.population]
        fitness_list = keras.utils.normalize(fitness_list)[0]
        for i, e in enumerate(self.population):
            e.fitness = fitness_list[i]

    def log_reward(self, path, generation=0, reward=0):
        with open(path, 'a') as file:
            file.write(f'{generation},{reward}\n')

    def create_population(self, size=1):
        population = []
        for i in tqdm(range(size), unit='population element'):
            population.append(NeatAnn(input_shape=self.input_shape, n_actions=self.env.action_space.n))

        return population

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
            del self.population[i].model
            self.population[i] = children[i]

    def save_best_element(self, save_as):
        if self.is_population_encoded:
            self.decode()

        best = sorted(self.population, key=lambda x: x.fitness)[-1]
        best.model.save(save_as)
