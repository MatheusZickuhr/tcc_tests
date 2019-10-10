import random

import cv2
import keras
import numpy as np
from keras.engine.saving import load_model
from ple import PLE
from ple.games import Catcher, MonsterKong

input_shape = (14,)


def resize_and_normalize_img(img):
    img = cv2.resize(img, input_shape[:2], interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return keras.utils.normalize(img.astype(np.float32))


def normalize_game_state(game_state):
    normalized_state = []
    for key in sorted(game_state.keys()):
        normalized_state.append(game_state[key])
    return keras.utils.normalize(normalized_state)


game = MonsterKong()
env = PLE(game, display_screen=True, force_fps=False)
env.init()
actions = env.getActionSet()

model = load_model('models\\mk_1.model')

while True:
    if env.game_over():
        env.reset_game()
    state = env.getGameState()
    print(model.predict(normalize_game_state(state)))
    action = np.argmax(model.predict(normalize_game_state(state)))
    r = env.act(actions[action])
