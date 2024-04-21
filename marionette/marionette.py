import numpy as np
import cv2
import mediapipe as mp
from vector import vector_from_points, calculate_angle
import logging
from gpiozero import Motor, PWMOutputDevice

class MotorController:
    def __init__(self, forward_pin=17, backward_pin=27, pwm_pin=18, pwm_frequency=1000):
        self.motor = Motor(forward=forward_pin, backward=backward_pin, pwm=True)
        self.pwm_device = PWMOutputDevice(pwm_pin, frequency=pwm_frequency)
        self.current_speed = 0

    def move_forward(self, speed):
        self.current_speed = speed
        self.motor.forward(speed)
        self.pwm_device.value = speed
        logging.info(f"Motor moving forward at speed {speed}")

    def move_backward(self, speed):
        self.current_speed = speed
        self.motor.backward(speed)
        self.pwm_device.value = speed
        logging.info(f"Motor moving backward at speed {speed}")

    def stop_motor(self):
        self.current_speed = 0
        self.motor.stop()
        self.pwm_device.value = 0
        logging.info("Motor stopped")

class PoseDetector:
    def __init__(self, motor_controller):
        self.motor_controller = motor_controller
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=0, smooth_landmarks=False,
            enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.drawing_utils = mp.solutions.drawing_utils

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logging.error("Cannot open webcam")
            return

        try:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    logging.warning("Ignoring empty camera frame.")
                    continue

                image = self.process_image(image)
                cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def process_image(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            self.draw_landmarks(image, results.pose_landmarks)
            angle = self.left_shoulder_angle(results.pose_world_landmarks)
            logging.info(f"Left shoulder angle: {np.degrees(angle)} degrees")
            self.control_motor(angle)
        return image

    def draw_landmarks(self, image, landmarks):
        self.drawing_utils.draw_landmarks(
            image, landmarks, mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.drawing_utils.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            connection_drawing_spec=self.drawing_utils.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

    def left_shoulder_angle(self, landmarks):
        left_shoulder = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]
        left_hip = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]
        
        left_upper_arm = vector_from_points(left_shoulder, left_elbow)
        left_shoulder_to_hip = vector_from_points(left_shoulder, left_hip)
        
        return calculate_angle(left_shoulder_to_hip, left_upper_arm)

    def control_motor(self, angle):
        speed = np.clip(np.sqrt(1.5 * (abs((angle - np.pi / 2) / (np.pi / 2)))), 0, 1)
        if angle < np.pi / 2:
            self.motor_controller.move_backward(speed)
        else:
            self.motor_controller.move_forward(speed)

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    motor_controller = MotorController()
    pose_detector = PoseDetector(motor_controller)
    pose_detector.run()

if __name__ == '__main__':
    main()
