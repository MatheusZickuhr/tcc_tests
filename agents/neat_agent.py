import random

from agents import agent_actions
from agents import base_agent
from bombman import Game
from utils import Timer


class NeatAgent(base_agent.Agent):

    current_element_index = -1
    population = None
    number_of_wins = 0
    number_of_kills = 0

    def start(self):
        self.current_element_index = -1
        self.population = self.create_population()

        print 'neat agent started'
        while self.has_next_element():
            current = self.get_next_element()
            time_alive_timer = Timer()
            self.play(current=current)
            print 'after play'
            self.calculate_fitness(element=current, time_alive=time_alive_timer.get_time_passed())

        self.mutate_population()

    def play(self, current):
        self.wait_for_game_start()

        print 'agent started playing'
        while not self.get_player_instance().is_dead():
            action = current.get_next_action()
            self.do_action(action)
        print 'agent finished playing'

    def create_population(self):
        population = []
        for i in range(10):
            population.append(Ann())

        return population

    def mutate_population(self):
        pass

    def crossover(self, element1, element2):
        pass

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
        return True if self.current_element_index < len(self.population) else False

    def wait_for_game_start(self):
        print 'waiting for game start to start playing'
        while not self.is_game_running() or self.is_player_dead():
            pass
        print 'game started'

    def is_game_running(self):
        return True if self.game_instance.state == Game.STATE_PLAYING else False


class Ann:

    def __init__(self, ann=None):
        self.ann = ann
        self.fitness = 0

    def get_next_action(self):
        return random.choice(agent_actions.possible_actions)
