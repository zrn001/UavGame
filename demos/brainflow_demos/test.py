from pynput.keyboard import Key, Controller
import time

keyboard = Controller()
time.sleep(3)
keyboard.press(Key.up)
print(1)