import random

import cv2
import keras
import numpy as np
from keras.engine.saving import load_model
from ple import PLE
from ple.games import Catcher, MonsterKong

input_shape = (50, 50, 3)


def resize_and_normalize_img(img):
    img = cv2.resize(img, input_shape[:2], interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return keras.utils.normalize(img.astype(np.float32))


def normalize_game_state(game_state):
    normalized_state = []
    for key in sorted(game_state.keys()):
        normalized_state.append(game_state[key])
    return keras.utils.normalize(normalized_state).reshape(input_shape)


game = MonsterKong()
env = PLE(game, display_screen=True, force_fps=False)
env.init()
actions = env.getActionSet()

positive_rewards = 0
total_rewards = 0
while True:
    if env.game_over():
        env.reset_game()
    r = env.act(None)

    print(env.getGameState())
