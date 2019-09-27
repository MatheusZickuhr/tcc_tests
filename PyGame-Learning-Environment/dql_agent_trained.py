from ple import PLE
from ple.games import FlappyBird

import random
from threading import Thread

import cv2
from keras.engine.saving import load_model
from ple.games.monsterkong import MonsterKong
import numpy as np

default_input_shape = (10, 10, 3)

game = FlappyBird()
p = PLE(game, display_screen=True, force_fps=False)
p.init()
actions = p.getActionSet()


def get_resized_image(img):
    img = cv2.resize(img, dsize=(50, 50), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32)


model = load_model('my_model.model')

done = False
while True:
    state = get_resized_image(p.getScreenRGB())
    cv2.imshow('game image', state)
    r = p.act(actions[np.argmax(model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0])])
    if p.game_over():
        p.reset_game()
