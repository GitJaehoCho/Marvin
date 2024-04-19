from gpiozero import Motor, Button, PWMOutputDevice
import time
import math

# Setup motor with PWM control
motor = Motor(forward=17, backward=27, pwm=True)
pwm_device = PWMOutputDevice(18, frequency=1000)  # PWM on pin 18 with 1 kHz frequency

# Setup encoder using gpiozero
encoder_a = Button(22, pull_up=True)
encoder_b = Button(23, pull_up=True)

# Constants
cpr = 48  # Counts per revolution

# Global variables
encoder_position = 0
current_angle = 0  # Current angle in radians
desired_angle = 0  # Desired angle in radians

def update_encoder_position():
    global encoder_position
    # Simulated encoder behavior based on the states of encoder A and B
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

def control_motor():
    global current_angle, desired_angle, encoder_position
    target_position = int((desired_angle * cpr) / (2 * math.pi))
    
    if encoder_position < target_position:
        motor.forward(0.5)  # Set speed to 50%
        pwm_device.value = 0.5  # Update PWM duty cycle
    elif encoder_position > target_position:
        motor.backward(0.5)  # Set speed to 50%
        pwm_device.value = 0.5  # Update PWM duty cycle
    else:
        motor.stop()
        pwm_device.value = 0  # Turn off PWM

    # Update current angle based on encoder position
    current_angle = (encoder_position * 2 * math.pi) / cpr

def main():
    global desired_angle  # Declare desired_angle as global to modify it
    try:
        last_a_state = encoder_a.is_pressed  # Store the last state of encoder A

        while True:
            # Check for change in encoder state
            if encoder_a.is_pressed != last_a_state:
                last_a_state = encoder_a.is_pressed
                update_encoder_position()
            
            # Simulate updating desired angle
            desired_angle += 0.01  # Increment desired angle continuously
            if desired_angle > 2 * math.pi:  # Reset after one full rotation
                desired_angle = 0

            control_motor()
            time.sleep(0.05)  # Control loop frequency

    finally:
        motor.stop()
        pwm_device.value = 0  # Ensure PWM is turned off when stopping

if __name__ == "__main__":
    main()
