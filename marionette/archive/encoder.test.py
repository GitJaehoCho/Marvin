import logging
from gpiozero import Button
from time import sleep

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Setup encoder using gpiozero
encoder_a = Button(22, pull_up=True)
encoder_b = Button(23, pull_up=True)

def monitor_encoder():
    last_a_state = encoder_a.is_pressed
    last_b_state = encoder_b.is_pressed
    count = 0

    logging.info("Starting to monitor the encoder. Rotate the encoder to check readings.")

    try:
        while True:
            a_state = encoder_a.is_pressed
            b_state = encoder_b.is_pressed

            if a_state != last_a_state or b_state != last_b_state:
                last_a_state = a_state
                last_b_state = b_state
                count += 1
                logging.info(f"Change detected: A={a_state}, B={b_state}, Count={count}")

            sleep(0.1)  # Check every 10ms

    except KeyboardInterrupt:
        logging.info("Stopped monitoring the encoder.")
        logging.info(f"Total changes detected: {count}")

if __name__ == "__main__":
    monitor_encoder()

