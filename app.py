import RPi.GPIO as gpio
import cv2
import os
from datetime import datetime
from sys import exit
from modules.source import Capture
from threading import Thread
from numpy import zeros
GPIO_READY_PIN = 17
GPIO_TRIGGER_PIN = 23
GPIO_OK_PIN = 27
GPIO_NG_PIN = 22
GPIO_LOGGING_PIN = 24
OK_THRESHOLD = 0.7
global_image = zeros((1, 1, 3), dtype='uint8')
def trigger_callback(channel):
    image = global_image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

def logging_callback(channel):
    ...

def check_template():
    global template
    path = 'template.jpg'
    if not os.path.exists(path):
        return False
    template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return True

def exit_routine():
    gpio.cleanup()

def setup():
    global capture
    gpio.setmode(gpio.BOARD)
    gpio.setup(GPIO_READY_PIN, gpio.OUT)
    gpio.setup(GPIO_TRIGGER_PIN, gpio.IN, pull_up_down=gpio.PUD_DOWN)
    gpio.setup(GPIO_OK_PIN, gpio.OUT)
    gpio.setup(GPIO_NG_PIN, gpio.OUT)
    gpio.setup(GPIO_LOGGING_PIN, gpio.IN, pull_up_down=gpio.PUD_DOWN)
    gpio.add_event_detect(GPIO_TRIGGER_PIN, gpio.RISING, callback=trigger_callback, bouncetime=200)
    gpio.add_event_detect(GPIO_LOGGING_PIN, gpio.RISING, callback=logging_callback, bouncetime=200)
    gpio.output(GPIO_OK_PIN, gpio.LOW)
    gpio.output(GPIO_NG_PIN, gpio.LOW)
    gpio.output(GPIO_READY_PIN, gpio.HIGH if check_template() else gpio.LOW)
    capture = Capture('', 0)
    capture.start()

def loop():
    global global_image
    global_image = capture.get_frame(None)
    cv2.imshow('', global_image)

if __name__ == '__main__':
    setup()
    try:
        while True:
            loop()
    except KeyboardInterrupt:
        exit_routine()