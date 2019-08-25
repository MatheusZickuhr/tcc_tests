import random

from agents import agent_actions
from agents import base_agent
from bombman import Game


class NeatAgent(base_agent.Agent):

    current_element_index = -1
    population = None

    def start(self):
        self.current_element_index = -1
        self.population = self.create_population()

        print 'neat agent started'
        while self.has_next_element():
            current = self.get_next_element()
            self.play(current=current)
            self.calculate_fitness(current)

        self.mutate_population()

    def play(self, current):
        self.wait_for_game_start()

        print 'agent started playing'
        while self.is_game_running():
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

    def calculate_fitness(self, element):
        pass

    def get_next_element(self):
        self.current_element_index += 1
        return self.population[self.current_element_index]

    def has_next_element(self):
        return True if self.current_element_index < len(self.population) else False

    def wait_for_game_start(self):
        print 'waiting for game start to start playing'
        while self.game_instance.state != Game.STATE_PLAYING:
            pass
        print 'game started'

    def is_game_running(self):
        return True if self.game_instance.state == Game.STATE_PLAYING else False


class Ann:

    def __init__(self, ann=None):
        self.ann = ann

    def get_next_action(self):
        return random.choice(agent_actions.possible_actions)
