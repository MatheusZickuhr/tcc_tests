import math
import random

from keras.models import load_model

from agents import base_agent
from agents import screen_reader
from agents.neat_ann import Ann
from bombman import Game
from utils import Timer


class NeatAgent(base_agent.Agent):
    current_element_index = -1
    population = None
    number_of_wins = 0
    number_of_kills = 0

    def start(self):
        self.wait_game_to_start()

        self.population = self.create_population(size=int(self.get_total_games_count() / 4))

        print 'neat agent started'

        while self.get_current_game() < self.get_total_games_count():

            self.current_element_index = -1

            while self.has_next_element():
                current = self.get_next_element()
                time_alive_timer = Timer()
                self.play(current=current)
                self.calculate_fitness(element=current, time_alive=time_alive_timer.get_time_passed())

            self.crossover_mutate_replace()

        self.save_best_element()

    def play_best(self):
        self.wait_game_to_start()

        current = self.load_best_ann()

        print 'neat agent started (playng best)'

        while self.get_current_game() < self.get_total_games_count():
            time_alive_timer = Timer()
            self.play(current=current)
            self.calculate_fitness(element=current, time_alive=time_alive_timer.get_time_passed())

    def play(self, current):
        self.wait_agent_can_play()

        print 'agent started playing'
        while not self.get_player_instance().is_dead():
            img_array = screen_reader.get_next_frame()
            action = current.get_next_action(img_array)
            print 'agent next action is ' + str(action)
            self.do_action(action)
        print 'agent finished playing'

    def create_population(self, size=1):
        population = []
        for i in range(1):
            population.append(Ann())

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

            random_number = math.random()

            for j in len(self.population):

                element = self.population[j]

                if random_number < sum([e.fitness / total_fitness for e in self.population[:j]]):
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
        print 'fitness calc started '
        fitness = 100 if self.get_player_instance().get_wins() > self.number_of_wins else 1
        fitness *= time_alive
        kills_dif = self.get_player_instance().get_kills() - self.number_of_kills
        fitness *= kills_dif if kills_dif > 0 else 1

        # atualiza os valores para o proxima rodada
        self.number_of_kills = self.get_player_instance().get_kills()
        self.number_of_wins = self.get_player_instance().get_wins()

        element.fitness = fitness

        print "fiteness {}".format(fitness)

    def get_next_element(self):
        self.current_element_index += 1
        return self.population[self.current_element_index]

    def has_next_element(self):
        return True if self.current_element_index + 1 < len(self.population) else False

    def wait_agent_can_play(self):
        print 'waiting until agent can play'
        while not self.is_game_running() or self.is_player_dead():
            pass
        print 'game started'

    def wait_game_to_start(self):
        print 'waiting for game to start'
        while self.get_total_games_count() is None or self.get_current_game() is None:
            pass

    def is_game_running(self):
        return True if self.game_instance.state == Game.STATE_PLAYING else False

    def save_best_element(self):
        best = sorted(self.population, key=lambda x: x.fitness)[-1]
        best.model.save('neat_agent_best.h5')
        print 'best element with fitness {}'.format(best.fitness)

    def load_best_ann(self):
        ann = Ann(is_child=True)
        ann.model = load_model('neat_agent_best.h5')
        return ann
