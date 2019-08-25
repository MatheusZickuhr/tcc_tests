import virtual_joystick
import agent_actions


class Agent:

    def __init__(self, game_instance=None):
        self.game_instance = game_instance

    def do_action(self, action):
        if action == agent_actions.GO_RIGHT:
            virtual_joystick.go_right()
        elif action == agent_actions.GO_LEFT:
            virtual_joystick.go_left()
        elif action == agent_actions.GO_UP:
            virtual_joystick.go_up()
        elif action == agent_actions.GO_DOWN:
            virtual_joystick.go_down()
        elif action == agent_actions.DROP_BOMB:
            virtual_joystick.drop_bomb()
