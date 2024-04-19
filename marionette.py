import numpy as np
import cv2
import mediapipe as mp
from gpiozero import Motor, Button, PWMOutputDevice

from vector import *

# Setup MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, smooth_landmarks=True,
                    enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawing_utils = mp.solutions.drawing_utils

# Motor setup
motor = Motor(forward=17, backward=27, pwm=True)
pwm_device = PWMOutputDevice(18, frequency=1000)

# Encoder setup using gpiozero
encoder_a = Button(22, pull_up=True)
encoder_b = Button(23, pull_up=True)

# Constants and global variables
cpr = 48  # Counts per revolution
encoder_position = 0
current_angle = 0  # Current angle in radians
desired_angle = 0  # Desired angle in radians (to be updated from pose)

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

def control_motor():
    global current_angle, desired_angle, encoder_position
    target_position = int((desired_angle * cpr) / (2 * np.pi))
    
    if encoder_position < target_position:
        motor.forward(0.5)
        pwm_device.value = 0.5
    elif encoder_position > target_position:
        motor.backward(0.5)
        pwm_device.value = 0.5
    else:
        motor.stop()
        pwm_device.value = 0

    current_angle = (encoder_position * 2 * np.pi) / cpr

def process_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        drawing_utils.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_utils.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            connection_drawing_spec=drawing_utils.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        angle_radians = left_shoulder_angle(results.pose_world_landmarks)
        global desired_angle
        desired_angle = angle_radians  # Set desired angle based on detected shoulder angle

    return image

def left_shoulder_angle(landmarks):
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    
    left_upper_arm = vector_from_points(left_shoulder, left_elbow)
    left_shoulder_to_hip = vector_from_points(left_shoulder, left_hip)
    angle = calculate_angle(left_shoulder_to_hip, left_upper_arm)
    return angle

def run_pose_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Cannot open webcam")

    try:
        while True:
            success, image = cap.read()
            if not success:
                continue

            image = process_image(image)
            cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break

            update_encoder_position()
            control_motor()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        motor.stop()
        pwm_device.value = 0

if __name__ == '__main__':
    run_pose_detection()
