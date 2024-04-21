import numpy as np
import cv2
import mediapipe as mp
from vector import vector_from_points, calculate_angle
import logging
from gpiozero import Motor, PWMOutputDevice
import time

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
        self.motor_paused = False  # Attribute to control motor state

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logging.error("Cannot open webcam")
            return

        prev_frame_time = time.time()
        fps = 0

        try:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    logging.warning("Ignoring empty camera frame.")
                    continue

                current_frame_time = time.time()
                duration = current_frame_time - prev_frame_time
                prev_frame_time = current_frame_time
                if duration > 0:
                    fps = round(1 / duration, 2)

                image = self.process_image(image)
                  
                flipped_image = cv2.flip(image, 1)
                
                # Display FPS on frame
                cv2.putText(flipped_image, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                
                cv2.imshow('MediaPipe Pose', flipped_image)
                
                key = cv2.waitKey(5)
                if key == 27:  # ESC key to break the loop
                    break
                if key == 32:  # Space bar to toggle pause
                    self.motor_paused = not self.motor_paused
                    if self.motor_paused:
                        self.motor_controller.stop_motor()
                        logging.info("Motor paused")
                    else:
                        logging.info("Motor resumed")
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
            if not self.motor_paused:
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
