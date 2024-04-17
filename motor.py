import RPi.GPIO as GPIO
import time
import threading
import math

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)  # Motor IN1
GPIO.setup(27, GPIO.OUT)  # Motor IN2
GPIO.setup(22, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Encoder A
GPIO.setup(23, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Encoder B
GPIO.setup(18, GPIO.OUT)  # PWM pin for EN_A

# Initialize PWM on EN_A
pwm = GPIO.PWM(18, 1000)  # Set PWM frequency to 1 kHz
pwm.start(20)  # Start PWM at 50% duty cycle for moderate speed

# Global variables
encoder_position = 0
current_angle = 0  # Current angle in radians
desired_angle = 0  # Desired angle in radians
cpr = 48  # Counts per revolution
lock = threading.Lock()

# Interrupt callback for encoder
def encoder_callback(channel):
    global encoder_position
    if GPIO.input(22):
        if GPIO.input(23):
            encoder_position += 1
        else:
            encoder_position -= 1
    else:
        if not GPIO.input(23):
            encoder_position += 1
        else:
            encoder_position -= 1

GPIO.add_event_detect(22, GPIO.BOTH, callback=encoder_callback)

def motor_forward():
    GPIO.output(17, GPIO.HIGH)
    GPIO.output(27, GPIO.LOW)

def motor_reverse():
    GPIO.output(17, GPIO.LOW)
    GPIO.output(27, GPIO.HIGH)

def motor_stop():
    pwm.ChangeDutyCycle(0)
    GPIO.output(17, GPIO.LOW)
    GPIO.output(27, GPIO.LOW)

# Thread to simulate receiving live input of radians
def input_simulation():
    global desired_angle
    while True:
        # Simulate a new desired angle in radians
        with lock:
            desired_angle += 3
        time.sleep(0.0001)  # Adjust frequency of new inputs here

# Thread to control motor based on desired angle
def motor_control():
    global current_angle, encoder_position, desired_angle
    while True:
        with lock:
            target_position = int((desired_angle * cpr) / (2 * math.pi))
        
        if encoder_position < target_position:
            motor_forward()
        elif encoder_position > target_position:
            motor_reverse()
        else:
            motor_stop()
        
        # Update current angle based on encoder position
        with lock:
            current_angle = (encoder_position * 2 * math.pi) / cpr
        time.sleep(0.05)  # Control loop frequency

def main():
    threading.Thread(target=input_simulation).start()  # Start input simulation thread
    threading.Thread(target=motor_control).start()  # Start motor control thread

    try:
        while True:  # Main loop to keep the program running
            time.sleep(10)
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    main()
