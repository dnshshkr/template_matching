import lgpio as gpio
import cv2
import os
from datetime import datetime
from sys import exit
from modules.source import Capture
from threading import Thread
from numpy import zeros
NAME = 'Template Matching'
GPIO_READY_PIN = 17
GPIO_TRIGGER_PIN = 23
GPIO_OK_PIN = 27
GPIO_NG_PIN = 22
GPIO_LOGGING_PIN = 24
OK_THRESHOLD = 0.7
LOG_PATH = '/home/pi/template_matching/logs'
global_live_image = zeros((1, 1, 3), dtype='uint8')
global_triggered_image = zeros((1, 1, 3), dtype='uint8')
def trigger_callback(*args):
    global global_triggered_image
    print('Triggered')
    print(args)
    image = global_live_image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    global_triggered_image = image.copy()
    score = 1.0 - max(0, cv2.minMaxLoc(cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)[1]))
    status = score >= OK_THRESHOLD
    gpio.gpio_write(h, GPIO_OK_PIN, status)
    gpio.gpio_write(h, GPIO_NG_PIN, not status)
    print('Score:', score)

def logging_callback(*args):
    timestamp = args[3]
    name = f'{datetime.fromtimestamp(timestamp).strftime("%d/%m/%Y.%H:%M")}.jpg'
    os.makedirs(LOG_PATH, exist_ok=True)
    cv2.imwrite(f'{LOG_PATH}/{name}', global_triggered_image)
    print('Logged:', name)

def check_template():
    global template
    path = 'template.jpg'
    if not os.path.exists(path):
        return False
    template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return True

def save_template():
    cv2.imwrite('template.jpg', global_live_image)
    check_template()
    print('Template saved')

def exit_routine():
    gpio.gpiochip_close(h)
    cv2.destroyAllWindows()
    capture.stop()

def setup():
    global capture, h
    h = gpio.gpiochip_open(0)
    gpio.gpio_claim_output(h, GPIO_READY_PIN)
    gpio.gpio_claim_input(h, GPIO_TRIGGER_PIN, gpio.SET_PULL_DOWN)
    gpio.gpio_claim_output(h, GPIO_OK_PIN)
    gpio.gpio_claim_output(h, GPIO_NG_PIN)
    gpio.gpio_claim_input(h, GPIO_LOGGING_PIN, gpio.SET_PULL_DOWN)
    gpio.gpio_claim_alert(h, GPIO_TRIGGER_PIN, gpio.RISING_EDGE)
    gpio.callback(h, GPIO_TRIGGER_PIN, gpio.RISING_EDGE, trigger_callback)
    gpio.gpio_claim_alert(h, GPIO_LOGGING_PIN, gpio.RISING_EDGE)
    gpio.callback(h, GPIO_LOGGING_PIN, gpio.RISING_EDGE, logging_callback)
    gpio.gpio_write(h, GPIO_READY_PIN, True if check_template() else False)
    capture = Capture(NAME, 0)
    capture.start()

def loop():
    global global_live_image
    global_live_image = capture.get_frame(None)
    cv2.imshow(NAME, global_live_image)
    key = cv2.waitKey(1) & 0xff
    if key == ord('S'.lower()):
        Thread(target=save_template).start()

if __name__ == '__main__':
    setup()
    try:
        cv2.namedWindow(NAME, cv2.WINDOW_NORMAL)
        while True:
            loop()
    except KeyboardInterrupt:
        exit_routine()