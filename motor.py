import lgpio
import time
import math

class MotorController:
    def __init__(self, gpio_chip=0, cpr=48):
        """Initialize the motor controller."""
        self.h = lgpio.gpiochip_open(gpio_chip)
        self.encoder_position = 0
        self.last_a_state = None
        self.last_b_state = None
        self.cpr = cpr  # Counts per revolution
        self.initialize_gpio()
        
    def initialize_gpio(self):
        """Claim the GPIOs for motor control and encoder inputs."""
        lgpio.gpio_claim_output(self.h, 17)  # Motor IN1
        lgpio.gpio_claim_output(self.h, 27)  # Motor IN2
        lgpio.gpio_claim_input(self.h, 22)   # Encoder A
        lgpio.gpio_claim_input(self.h, 23)   # Encoder B
        lgpio.gpio_claim_output(self.h, 18)  # PWM pin for EN_A

    def update_encoder_position(self):
        """Update the encoder position by reading the encoder pins directly."""
        a_state = lgpio.gpio_read(self.h, 22)
        b_state = lgpio.gpio_read(self.h, 23)

        if self.last_a_state is not None and self.last_b_state is not None:
            if a_state != self.last_a_state or b_state != self.last_b_state:
                # Determine direction of rotation
                if (self.last_a_state and not self.last_b_state and a_state and b_state) or \
                   (self.last_a_state and self.last_b_state and not a_state and b_state) or \
                   (not self.last_a_state and self.last_b_state and not a_state and not b_state) or \
                   (not self.last_a_state and not self.last_b_state and a_state and not b_state):
                    self.encoder_position += 1
                else:
                    self.encoder_position -= 1

        self.last_a_state = a_state
        self.last_b_state = b_state

    def motor_control(self, direction):
        """Set motor direction."""
        if direction == 'forward':
            lgpio.gpio_write(self.h, 17, 1)
            lgpio.gpio_write(self.h, 27, 0)
        elif direction == 'reverse':
            lgpio.gpio_write(self.h, 17, 0)
            lgpio.gpio_write(self.h, 27, 1)
        else:
            self.motor_stop()

    def motor_stop(self):
        """Stop motor and PWM signal."""
        lgpio.tx_pwm(self.h, 18, 1000, 0)  # Stop PWM
        lgpio.gpio_write(self.h, 17, 0)
        lgpio.gpio_write(self.h, 27, 0)

    def run(self):
        """Main control loop."""
        lgpio.tx_pwm(self.h, 18, 1000, 60)  # 1 kHz frequency, 20% duty cycle
        desired_angle = 0  # Start at angle 0
        input_interval = 0.1  # Interval for updating desired angle
        last_input_time = time.time()

        try:
            while True:
                if time.time() - last_input_time >= input_interval:
                    desired_angle += math.radians(10)  # Increment angle by 10 degrees
                    last_input_time = time.time()

                self.update_encoder_position()
                target_position = int((desired_angle * self.cpr) / (2 * math.pi))
                if self.encoder_position < target_position:
                    self.motor_control('forward')
                elif self.encoder_position > target_position:
                    self.motor_control('reverse')
                else:
                    self.motor_stop()

                # Print current encoder position
                print(f"Current Encoder Position: {self.encoder_position}")
                
                time.sleep(0.05)  # Loop at 20 Hz

        except KeyboardInterrupt:
            self.motor_stop()
        finally:
            lgpio.gpiochip_close(self.h)

if __name__ == "__main__":
    mc = MotorController()
    mc.run()

