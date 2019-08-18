import base_agent
import agent_actions
import random


class DummyAgent(base_agent.Agent):

    def __init__(self):
        print('dummy agent created')

    def start(self):
        print('dummy agent started')
        while True:
            self.do_action(
                random.choice(agent_actions.possible_actions)
            )
