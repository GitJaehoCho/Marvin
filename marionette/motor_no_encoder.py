import time
import logging
from gpiozero import Motor, PWMOutputDevice

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Setup motor with PWM control
motor = Motor(forward=17, backward=27, pwm=True)  # motor driver
pwm_device = PWMOutputDevice(18, frequency=1000)

# Global variables
current_speed = 0

def move_forward():
    global current_speed
    current_speed = 0.5  # Set speed to half of the maximum
    motor.forward(current_speed)
    pwm_device.value = current_speed
    logging.info("Motor moving forward")

def move_backward():
    global current_speed
    current_speed = 0.5  # Set speed to half of the maximum
    motor.backward(current_speed)
    pwm_device.value = current_speed
    logging.info("Motor moving backward")

def stop_motor():
    global current_speed
    current_speed = 0
    motor.stop()
    pwm_device.value = current_speed
    logging.info("Motor stopped")

def main():
    try:
        while True:
            move_forward()
            time.sleep(2)  # Move forward for 2 seconds
            stop_motor()
            time.sleep(1)  # Pause for 1 second
            move_backward()
            time.sleep(2)  # Move backward for 2 seconds
            stop_motor()
            time.sleep(1)  # Pause for 1 second
    except KeyboardInterrupt:
        stop_motor()
        logging.info("Motor stopped due to keyboard interrupt")

if __name__ == "__main__":
    main()
