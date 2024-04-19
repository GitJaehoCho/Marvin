import time
import math
import logging
from gpiozero import Motor, Button, PWMOutputDevice

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Setup motor with PWM control
motor = Motor(forward=17, backward=27, pwm=True) # motor driver
pwm_device = PWMOutputDevice(18, frequency=1000)

# Setup encoder using gpiozero
encoder_a = Button(22, pull_up=True) # Quad Encoder A signal
encoder_b = Button(23, pull_up=True) # Quad Encoder B signal

# Constants
cpr = 48  # Counts per revolution

# Global variables
encoder_position = 0
current_angle = 0  # Current angle in radians
desired_angle = 0  # Desired angle in radians

def update_encoder_position():
    global encoder_position
    if encoder_a.is_pressed:
        if encoder_b.is_pressed:
            encoder_position += 1
        else:
            encoder_position -= 1
    else:
        if not encoder_b.is_pressed:
            encoder_position += 1
        else:
            encoder_position -= 1
    logging.debug(f"Encoder position updated: {encoder_position}")

def control_motor():
    global current_angle, desired_angle, encoder_position
    target_position = int((desired_angle * cpr) / (2 * math.pi))
    error = target_position - encoder_position
    tolerance = 1  # you can adjust this tolerance based on your precision needs
    kp = 0.1  # Proportional gain, this value may need tuning

    if abs(error) <= tolerance:
        motor.stop()
        pwm_device.value = 0
        logging.info(f"Motor stopped at position {encoder_position}")
    else:
        speed = kp * abs(error)
        speed = max(0.1, min(0.5, speed))  # Limit speed to between 0.1 and 0.5
        if error > 0:
            motor.forward(speed)
            pwm_device.value = speed
            logging.info(f"Motor moving forward: target {target_position}, current {encoder_position}, speed {speed}")
        else:
            motor.backward(speed)
            pwm_device.value = speed
            logging.info(f"Motor moving backward: target {target_position}, current {encoder_position}, speed {speed}")

    current_angle = (encoder_position * 2 * math.pi) / cpr
    logging.debug(f"Current angle: {math.degrees(current_angle)} degrees")

def set_desired_angle():
    global desired_angle
    angle_degrees = float(input("Enter desired angle in degrees: "))
    desired_angle = math.radians(angle_degrees)
    logging.info(f"Desired angle set to {desired_angle} radians ({angle_degrees} degrees)")

def main():
    try:
        while True:
            set_desired_angle()
            while True:
                update_encoder_position()
                control_motor()
                time.sleep(0.05)
                # Exit the loop when motor is close enough to the target
                if abs(encoder_position - int((desired_angle * cpr) / (2 * math.pi))) <= 1:
                    break
    finally:
        motor.stop()
        pwm_device.value = 0
        logging.info("Motor and PWM safely stopped")

if __name__ == "__main__":
    main()
