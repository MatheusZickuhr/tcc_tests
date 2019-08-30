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
        elif action == agent_actions.WAIT:
            virtual_joystick.wait()

    def get_player_instance(self):
        if self.game_instance.game_map is not None:
            return self.game_instance.game_map.players[0]
        else:
            return None

    def is_player_dead(self):
        if self.get_player_instance() is not None:
            return self.get_player_instance().is_dead()
        return True
