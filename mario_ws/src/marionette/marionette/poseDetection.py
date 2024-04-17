import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import cv2
import mediapipe as mp

from .vector import *

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, smooth_landmarks=True,
                    enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawing_utils = mp.solutions.drawing_utils

class PosePublisher(Node):
    def __init__(self):
        super().__init__('pose_publisher')
        self.publisher_ = self.create_publisher(Float64, 'target_angle', 10)
        self.timer = self.create_timer(0.1, self.run_pose_detection)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Cannot open webcam")

    def run_pose_detection(self):
        success, image = self.cap.read()
        if not success:
            self.get_logger().info('Ignoring empty camera frame.')
            return

        image = self.process_image(image)
        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        if cv2.waitKey(1) & 0xFF == 27:
            self.destroy_node()
            cv2.destroyAllWindows()
            self.cap.release()

    def process_image(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            self.draw_landmarks(image, results.pose_landmarks)
            angle = self.left_shoulder_angle(results.pose_world_landmarks)
            self.publisher_.publish(Float64(data=angle))
        return image

    def draw_landmarks(self, image, landmarks):
        drawing_utils.draw_landmarks(
            image, landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_utils.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            connection_drawing_spec=drawing_utils.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

    def left_shoulder_angle(self, landmarks):
        left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        left_upper_arm = vector_from_points(left_shoulder, left_elbow)
        left_shoulder_to_hip = vector_from_points(left_shoulder, left_hip)
        angle = calculate_angle(left_shoulder_to_hip, left_upper_arm)
        return angle

def main(args=None):
    rclpy.init(args=args)
    pose_publisher = PosePublisher()
    rclpy.spin(pose_publisher)

if __name__ == '__main__':
    main()
