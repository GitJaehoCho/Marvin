import numpy as np
import cv2
import mediapipe as mp
from vector import *
import logging
from gpiozero import Motor, PWMOutputDevice

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Setup motor with PWM control
motor = Motor(forward=17, backward=27, pwm=True)  # motor driver
pwm_device = PWMOutputDevice(18, frequency=1000)

# Global variables
current_speed = 0

def move_forward(speed):
    global current_speed
    current_speed = speed
    motor.forward(speed)
    pwm_device.value = speed
    logging.info("Motor moving forward at speed {}".format(speed))

def move_backward(speed):
    global current_speed
    current_speed = speed
    motor.backward(speed)
    pwm_device.value = speed
    logging.info("Motor moving backward at speed {}".format(speed))

def stop_motor():
    global current_speed
    current_speed = 0
    motor.stop()
    pwm_device.value = current_speed
    logging.info("Motor stopped")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=0, smooth_landmarks=True,
                    enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawing_utils = mp.solutions.drawing_utils

def run_pose_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Cannot open webcam")

    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                raise ValueError("Ignoring empty camera frame.")

            image = process_image(image)
            cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:  # Exit loop if 'ESC' is pressed
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

def process_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        draw_landmarks(image, results.pose_landmarks)
        angle = left_shoulder_angle(results.pose_world_landmarks)
        logging.info("Left shoulder angle: {:.2f} degrees".format(np.degrees(angle)))
        control_motor(angle)
    return image

def draw_landmarks(image, landmarks):
    drawing_utils.draw_landmarks(
        image, landmarks, mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=drawing_utils.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
        connection_drawing_spec=drawing_utils.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

def left_shoulder_angle(landmarks):
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER] # 11
    left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW] # 13
    left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP] # 23
    
    left_upper_arm = vector_from_points(left_shoulder, left_elbow)
    left_shoulder_to_hip = vector_from_points(left_shoulder, left_hip)

    angle = calculate_angle(left_shoulder_to_hip, left_upper_arm)
    return angle

def control_motor(angle):
    # Map angle to motor speed
    speed = min(1.0, max(0.0, angle / np.pi))  # Normalize angle to range [0, 1]
    if angle > np.pi / 2:  # Move backward if angle is greater than 90 degrees
        move_backward(speed)
    else:  # Move forward otherwise
        move_forward(speed)

def main():
    run_pose_detection()

if __name__ == '__main__':
    main()
