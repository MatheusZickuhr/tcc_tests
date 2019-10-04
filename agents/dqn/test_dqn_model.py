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


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # return the resized image
    return keras.utils.normalize(resized)


def normalize_game_state(game_state):
    normalized_state = []
    for key in sorted(game_state.keys()):
        normalized_state.append(game_state[key])
    return keras.utils.normalize(normalized_state).reshape(input_shape)


game = MonsterKong()
env = PLE(game, display_screen=True, force_fps=False)
env.init()
actions = env.getActionSet()
print(game.getScreenRGB().shape)
# model = load_model('models\\catcher_test.model')

positive_rewards = 0
total_rewards = 0
while True:
    if env.game_over():
        env.reset_game()
    r = env.act(random.choice(actions))
    print(r)
    # state = normalize_game_state(env.getGameState())
    # action = np.argmax(model.predict(np.array([state])))
    # reward = env.act(random.choice(actions))
    cv2.imshow('teste', resize_and_normalize_img(env.getScreenRGB()))

    # if reward == 1:
    #     positive_rewards += 1
    # if reward == -1 or reward == 1:
    #     total_rewards += 1
    #
    # if total_rewards > 0:
    #     print(positive_rewards / total_rewards)
