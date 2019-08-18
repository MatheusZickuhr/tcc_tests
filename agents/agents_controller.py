from pynput.keyboard import Key, Listener
from dummy_agent import DummyAgent


def on_key_pressed(key):
    pass


def on_key_release(key):
    if key == start_agent_key:
        print('agent started')
        agent = DummyAgent()
        agent.start()
        listener.stop()


listener = Listener(on_press=on_key_pressed, on_release=on_key_release)
start_agent_key = Key.f1


class AgentsController:

    def __init__(self):
        print('press {} to start the agent'.format(start_agent_key))
        listener.start()
