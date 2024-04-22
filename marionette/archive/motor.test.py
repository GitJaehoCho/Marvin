import logging
from gpiozero import Button
import math

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Setup encoder using gpiozero with callbacks for both rising and falling edges
encoder_a = Button(22, pull_up=True)  # Quad Encoder A signal
encoder_b = Button(23, pull_up=True)  # Quad Encoder B signal

# Constants
cpr = 240  # Counts per revolution, adjust as per your encoder specification

# Global variables
encoder_position = 0

def encoder_callback():
    global encoder_position
    a = encoder_a.is_pressed
    b = encoder_b.is_pressed
    if a and b or not a and not b:
        encoder_position += 1
    else:
        encoder_position -= 1
    current_angle_degrees = (encoder_position * 360.0) / cpr
    logging.info(f"Current angle: {current_angle_degrees:.2f} degrees")

# Attach callbacks to encoder signal changes
encoder_a.when_pressed = encoder_callback
encoder_a.when_released = encoder_callback
encoder_b.when_pressed = encoder_callback
encoder_b.when_released = encoder_callback

# Keep the program running
try:
    import signal
    signal.pause()  # This will keep the program running until a keyboard interrupt
except KeyboardInterrupt:
    pass  # Cleanly handle the keyboard interrupt
finally:
    logging.info("Program terminated")
