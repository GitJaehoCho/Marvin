import numpy as np
import cv2
import mediapipe as mp

def vector_from_points(p1, p2):
    """Create a vector from two points."""
    return np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])

def calculate_angle(v1, v2):
    """Calculate the angle between two vectors."""
    dot_product = np.dot(v1, v2)
    magnitude_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_angle = dot_product / magnitude_product
    return np.arccos(cos_angle)

def project_vector_onto_plane(vector, plane_normal):
    """Project a vector onto a plane defined by its normal vector."""
    plane_normal_normalized = plane_normal / np.linalg.norm(plane_normal)
    dot_product = np.dot(vector, plane_normal_normalized)
    return vector - dot_product * plane_normal_normalized

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, smooth_landmarks=True,
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
        print(left_shoulder_angle(results.pose_world_landmarks))
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

def main():
    run_pose_detection()
if __name__ == '__main__':
    main()