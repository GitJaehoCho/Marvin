import RPi.GPIO as GPIO
import time
import threading
import math
import numpy as np
import cv2
import mediapipe as mp

# GPIO Setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)  # Motor IN1
GPIO.setup(27, GPIO.OUT)  # Motor IN2
GPIO.setup(22, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Encoder A
GPIO.setup(23, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Encoder B
GPIO.setup(18, GPIO.OUT)  # PWM pin for EN_A

# Initialize PWM
pwm = GPIO.PWM(18, 1000)
pwm.start(20)

# Global variables
encoder_position = 0
current_angle = 0
desired_angle = 0
cpr = 48
lock = threading.Lock()

# Motor control functions
def motor_forward():
    GPIO.output(17, GPIO.HIGH)
    GPIO.output(27, GPIO.LOW)

def motor_reverse():
    GPIO.output(17, GPIO.LOW)
    GPIO.output(27, GPIO.HIGH)

def motor_stop():
    pwm.ChangeDutyCycle(0)
    GPIO.output(17, GPIO.LOW)
    GPIO.output(27, GPIO.LOW)

# Encoder callback
def encoder_callback(channel):
    global encoder_position
    if GPIO.input(22):
        if GPIO.input(23):
            encoder_position += 1
        else:
            encoder_position -= 1
    else:
        if not GPIO.input(23):
            encoder_position += 1
        else:
            encoder_position -= 1
GPIO.add_event_detect(22, GPIO.BOTH, callback=encoder_callback)

# Motor control thread
def motor_control():
    global current_angle, encoder_position, desired_angle
    while True:
        with lock:
            target_position = int((desired_angle * cpr) / (2 * math.pi))
        if encoder_position < target_position:
            motor_forward()
        elif encoder_position > target_position:
            motor_reverse()
        else:
            motor_stop()
        with lock:
            current_angle = (encoder_position * 2 * math.pi) / cpr
        time.sleep(0.05)

# Pose detection setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.5)

# Pose processing functions
def vector_from_points(p1, p2):
    return np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])

def calculate_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitude_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_angle = dot_product / magnitude_product
    return np.arccos(cos_angle)

# Main function for pose detection
def run_pose_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Cannot open webcam")
    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            if results.pose_landmarks:
                left_shoulder = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                left_elbow = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
                left_hip = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                left_upper_arm = vector_from_points(left_shoulder, left_elbow)
                left_shoulder_to_hip = vector_from_points(left_shoulder, left_hip)
                angle = calculate_angle(left_shoulder_to_hip, left_upper_arm)
                with lock:
                    desired_angle = angle  # Update the desired angle directly from pose detection
            cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

# Initialize and run all threads
def main():
    threading.Thread(target=run_pose_detection).start()
    threading.Thread(target=motor_control).start()
    try:
        while True:
            time.sleep(10)
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    main()
