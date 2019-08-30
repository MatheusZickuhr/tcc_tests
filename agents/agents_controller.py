from threading import Thread

from pynput.keyboard import Key, Listener

from agents.dummy_agent import DummyAgent
from agents.neat_agent import NeatAgent

start_agent_key = Key.f1


class AgentsController:

    def __init__(self, game_instance=None):
        self.listener = Listener(on_press=self.on_key_pressed, on_release=self.on_key_release)
        self.game_instance = game_instance
        print('press {} to start the agent'.format(start_agent_key))
        self.listener.start()

    def on_key_pressed(self, key):
        pass

    def on_key_release(self, key):
        def start_agent():
            print('agent started')
            agent = NeatAgent(game_instance=self.game_instance)
            agent.start()

        if key == start_agent_key:
            Thread(target=start_agent).start()
            self.listener.stop()
