import argparse
import sys
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

class PoseLandmarker:
    def __init__(self, model, num_poses, min_pose_detection_confidence,
                 min_pose_presence_confidence, min_tracking_confidence,
                 output_segmentation_masks, camera_id, width, height):
        self.model = model
        self.num_poses = num_poses
        self.min_pose_detection_confidence = min_pose_detection_confidence
        self.min_pose_presence_confidence = min_pose_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.output_segmentation_masks = output_segmentation_masks
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
        self.detector = None
        self.FPS = 0
        self.COUNTER = 0
        self.START_TIME = time.time()
        self.DETECTION_RESULT = None

    def start_camera(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        print("Camera started.")

    def initialize_detector(self):
        base_options = python.BaseOptions(model_asset_path=self.model)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_poses=self.num_poses,
            min_pose_detection_confidence=self.min_pose_detection_confidence,
            min_pose_presence_confidence=self.min_pose_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_segmentation_masks=self.output_segmentation_masks,
            result_callback=self.save_result)
        self.detector = vision.PoseLandmarker.create_from_options(options)
        print("Detector initialized.")

    def save_result(self, result, unused_output_image, timestamp_ms):
        if self.COUNTER % 10 == 0:
            self.FPS = 10 / (time.time() - self.START_TIME)
            self.START_TIME = time.time()

        self.DETECTION_RESULT = result
        self.COUNTER += 1

    def run_detection(self):
        if not self.cap:
            self.start_camera()
        if not self.detector:
            self.initialize_detector()

        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                sys.exit('ERROR: Unable to read from webcam. Please verify your webcam settings.')
            image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            self.detector.detect_async(mp_image, time.time_ns() // 1_000_000)
            self.display_results(image)

            if cv2.waitKey(1) == 27:  # ESC key
                break

    def display_results(self, image):
        fps_text = f'FPS = {self.FPS:.1f}'
        cv2.putText(image, fps_text, (24, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
        
        if self.DETECTION_RESULT and self.DETECTION_RESULT.pose_landmarks:
            for pose_landmarks in self.DETECTION_RESULT.pose_landmarks:
                # Draw the pose landmarks.
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                    for landmark in pose_landmarks
                ])
                mp.solutions.drawing_utils.draw_landmarks(
                    image,
                    pose_landmarks_proto,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_pose_landmarks_style())
        
        if self.output_segmentation_masks and self.DETECTION_RESULT:
            if self.DETECTION_RESULT.segmentation_masks is not None:
                segmentation_mask = self.DETECTION_RESULT.segmentation_masks[0].numpy_view()
                mask_color = (100, 100, 0)  # cyan
                overlay_alpha = 0.5
                mask_image = np.zeros(image.shape, dtype=np.uint8)
                mask_image[:] = mask_color
                condition = np.stack((segmentation_mask,) * 3, axis=-1) > 0.1
                visualized_mask = np.where(condition, mask_image, image)
                image = cv2.addWeighted(image, 1 - overlay_alpha, visualized_mask, overlay_alpha, 0)
        
        cv2.imshow('pose_landmarker', image)

    def close(self):
        self.detector.close()
        self.cap.release()
        cv2.destroyAllWindows()
        print("Resources released.")

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', default='pose_landmarker.task', help='Name of the pose landmarker model bundle.')
    parser.add_argument('--numPoses', type=int, default=1, help='Max number of poses that can be detected.')
    parser.add_argument('--minPoseDetectionConfidence', type=float, default=0.5, help='Minimum confidence for pose detection.')
    parser.add_argument('--minPosePresenceConfidence', type=float, default=0.5, help='Minimum confidence for pose presence.')
    parser.add_argument('--minTrackingConfidence', type=float, default=0.5, help='Minimum confidence for tracking.')
    parser.add_argument('--outputSegmentationMasks', action='store_true', help='Output segmentation masks.')
    parser.add_argument('--cameraId', type=int, default=0, help='ID of the camera to use.')
    parser.add_argument('--frameWidth', type=int, default=1280, help='Width of the camera frame.')
    parser.add_argument('--frameHeight', type=int, default=960, help='Height of the camera frame.')
    args = parser.parse_args()

    landmarker = PoseLandmarker(args.model, args.numPoses, args.minPoseDetectionConfidence,
                                args.minPosePresenceConfidence, args.minTrackingConfidence,
                                args.outputSegmentationMasks, args.cameraId, args.frameWidth, args.frameHeight)
    landmarker.run_detection()
    landmarker.close()

if __name__ == '__main__':
    main()
