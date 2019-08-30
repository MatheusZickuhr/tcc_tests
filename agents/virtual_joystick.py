from pynput.keyboard import Key, Controller
from threading import Thread
import time

keyboard = Controller()


def hold_key(key=None, for_time=None):
    def hold():
        keyboard.press(key)
        time.sleep(for_time)
        keyboard.release(key)

    thread = Thread(target=hold)
    thread.start()
    thread.join()


def go_up():
    hold_key(key='w', for_time=0.5)


def go_down():
    hold_key(key='s', for_time=0.5)


def go_left():
    hold_key(key='a', for_time=0.5)


def go_right():
    hold_key(key='d', for_time=0.5)


def drop_bomb():
    keyboard.press('c')
    keyboard.release('c')


def wait():
    time.sleep(0.5)
