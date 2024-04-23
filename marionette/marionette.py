import numpy as np
import cv2
import mediapipe as mp
import logging
from gpiozero import Motor, PWMOutputDevice
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MotorController:
    def __init__(self, forward_pin=17, backward_pin=27, pwm_pin=18, pwm_frequency=1000):
        self.motor = Motor(forward=forward_pin, backward=backward_pin, pwm=True)
        self.pwm_device = PWMOutputDevice(pwm_pin, frequency=pwm_frequency)
        self.current_speed = 0
        self.motor_state = "Waiting for Person"

    def move_forward(self, speed):
        if self.current_speed != speed:
            self.current_speed = speed
            self.motor.forward(speed)
            self.pwm_device.value = speed
            self.update_motor_state("Moving Forward")

    def move_backward(self, speed):
        if self.current_speed != speed:
            self.current_speed = speed
            self.motor.backward(speed)
            self.pwm_device.value = speed
            self.update_motor_state("Moving Backward")

    def stop_motor(self):
        if self.current_speed != 0:
            self.current_speed = 0
            self.motor.stop()
            self.pwm_device.value = 0
            self.update_motor_state("No Person")

    def update_motor_state(self, state):
        if self.motor_state != state:
            self.motor_state = state
            logging.info(f"Motor status: {state}")

    def get_motor_status(self):
        return self.motor_state

class PoseDetector:
    def __init__(self, motor_controller):
        self.motor_controller = motor_controller
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=0, smooth_landmarks=False,
            enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.drawing_utils = mp.solutions.drawing_utils
        self.motor_paused = False

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logging.error("Cannot open webcam")
            return

        prev_frame_time = time.time()
        frame_rate_calc = 0

        try:
            while True:
                success, image = cap.read()
                if not success:
                    logging.warning("Ignoring empty camera frame.")
                    continue

                current_frame_time = time.time()
                frame_rate_calc = (0.9 * frame_rate_calc) + (0.1 * (1 / (current_frame_time - prev_frame_time)))
                prev_frame_time = current_frame_time

                image = self.process_image(image)
                image = cv2.flip(image, 1)
                self.display_frame(image, frame_rate_calc)

                key = cv2.waitKey(5)
                if key == 27:  # ESC key to exit
                    break
                elif key == 32:  # Space bar to pause/resume
                    self.motor_paused = not self.motor_paused
                    if self.motor_paused:
                        self.motor_controller.stop_motor()
                        self.motor_controller.update_motor_state("Paused")
                    else:
                        self.motor_controller.update_motor_state("Resumed")

        finally:
            cap.release()
            cv2.destroyAllWindows()

    def process_image(self, image):
        if not self.motor_paused:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)
            self.draw_landmarks(image, results.pose_landmarks)
            self.handle_pose(results)
        return image

    def draw_landmarks(self, image, landmarks):
        if landmarks:
            self.drawing_utils.draw_landmarks(
                image, landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.drawing_utils.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                connection_drawing_spec=self.drawing_utils.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

    def handle_pose(self, results):
        if results.pose_landmarks and not self.motor_paused:
            angle = self.left_shoulder_angle(results.pose_world_landmarks)
            self.control_motor(angle)
        elif not results.pose_landmarks:
            self.motor_controller.stop_motor()

    def left_shoulder_angle(self, landmarks):
        left_shoulder = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]
        left_hip = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]

        upper_arm_vector = np.array([left_elbow.x - left_shoulder.x, left_elbow.y - left_shoulder.y, left_elbow.z - left_shoulder.z])
        shoulder_to_hip_vector = np.array([left_hip.x - left_shoulder.x, left_hip.y - left_shoulder.y, left_hip.z - left_shoulder.z])

        dot_product = np.dot(upper_arm_vector, shoulder_to_hip_vector)
        norm_product = np.linalg.norm(upper_arm_vector) * np.linalg.norm(shoulder_to_hip_vector)
        angle = np.arccos(dot_product / norm_product)
        return angle

    def control_motor(self, angle):
        speed = np.clip(np.sqrt(1.5 * (abs((angle - np.pi / 2) / (np.pi / 2)))), 0, 1)
        if angle < np.pi / 2:
            self.motor_controller.move_backward(speed)
        else:
            self.motor_controller.move_forward(speed)

    def display_frame(self, image, fps):
        cv2.putText(image, f'FPS: {fps:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        motor_status = self.motor_controller.get_motor_status()
        cv2.putText(image, motor_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('MediaPipe Pose', image)

def main():
    pose_detector = PoseDetector(MotorController())
    pose_detector.run()

if __name__ == '__main__':
    main()