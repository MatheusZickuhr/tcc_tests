import virtual_joystick
import agent_actions
from bombman import Game


class Agent:

    def __init__(self, game_instance=None):
        self.game_instance = game_instance

        self.initialize()

    def initialize(self):
        print 'initialize not implemented'
        pass

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

    def get_total_games_count(self):
        if self.game_instance is None or self.game_instance.game_map is None:
            return None

        return self.game_instance.game_map.max_games

    def get_current_game(self):
        if self.game_instance is None or self.game_instance.game_map is None:
            return None

        return self.game_instance.game_map.game_number

    def wait_game_to_start(self):
        print 'waiting for game to start'
        while self.get_total_games_count() is None or self.get_current_game() is None:
            pass

    def wait_agent_can_play(self):
        print 'waiting until agent can play'
        while not self.is_game_running() or self.is_player_dead():
            pass
        print 'game started'

    def is_game_running(self):
        return True if self.game_instance.state == Game.STATE_PLAYING else False
