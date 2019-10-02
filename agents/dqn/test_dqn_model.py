import cv2
import keras
import numpy as np
from keras.engine.saving import load_model
from ple import PLE
from ple.games import Catcher

input_shape = (10, 10, 3)


def resize_img(img):
    img = cv2.resize(img, dsize=input_shape[:2], interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return keras.utils.normalize(img.astype(np.float32))


game = Catcher()
env = PLE(game, display_screen=True, force_fps=False)
env.init()
actions = env.getActionSet()

model = load_model('models\\catcher_model.model')

while True:
    if env.game_over():
        env.reset_game()
    state = resize_img(env.getScreenRGB())
    action = np.argmax(model.predict(np.array([state])))
    reward = env.act(actions[action])

    if reward == 1:
        print('pegou')
    elif reward == -1:
        print('errou')
