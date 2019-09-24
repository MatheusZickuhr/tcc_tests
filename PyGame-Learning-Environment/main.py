from ple.games.monsterkong import MonsterKong
from ple import PLE
import random

game = MonsterKong()

p = PLE(game, display_screen=True, force_fps=True)
p.init()

nb_frames = 100000
reward = 0.0

for f in range(nb_frames):
    if p.game_over():  # check if the game is over
        p.reset_game()

    obs = p.getScreenRGB()
    actions = p.getActionSet()
    reward = p.act(actions[random.randint(0, len(actions) - 1)])
