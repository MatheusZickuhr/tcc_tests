import cv2
import keras
import numpy as np
from keras.engine.saving import load_model
from ple import PLE
from ple.games import FlappyBird

input_shape = (10, 10, 3)


def resize_img(img):
    img = cv2.resize(img, dsize=input_shape[:2], interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return keras.utils.normalize(img.astype(np.float32) / 255)


game = FlappyBird()
env = PLE(game, display_screen=True, force_fps=False)
env.init()
actions = env.getActionSet()

model = load_model('models\\dql_fb.model')

done = False
while True:
    if env.game_over():
        env.reset_game()

    state = resize_img(env.getScreenRGB())
    action = np.argmax(model.predict(np.array([state])))
    env.act(actions[action])
