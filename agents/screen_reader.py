import numpy as np
import cv2
from mss import mss
from PIL import Image

mon = {'top': 0, 'left': 0, 'width': 960, 'height': 720}

sct = mss()

OUTPUT_WIDTH = 96
OUTPUT_HEIGHT = 72


def get_next_frame():
    sct.get_pixels(mon)
    img = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
    img = img.resize((OUTPUT_WIDTH, OUTPUT_HEIGHT), Image.ANTIALIAS)
    img_array = np.array(img)
    return img_array
