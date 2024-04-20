# Copyright 2023 The MediaPipe Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main scripts to run pose landmarker using a class-based approach for easier integration."""

import argparse
import sys
import time
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

class PoseDetector:
    def __init__(self, model, num_poses, min_pose_detection_confidence, min_pose_presence_confidence, min_tracking_confidence, output_segmentation_masks, camera_id, width, height):
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
        self.latest_result = None
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

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

    def save_result(self, result, unused_output_image, timestamp_ms):
        self.latest_result = result

    def get_latest_result(self):
        return self.latest_result

    def stop(self):
        self.detector.close()
        self.cap.release()
        cv2.destroyAllWindows()

    def run_once(self):
        success, image = self.cap.read()
        if not success:
            sys.exit('ERROR: Unable to read from webcam. Please verify your webcam settings.')

        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        self.detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        # Display the image
        if self.latest_result:
            self.display_results(image)

        cv2.imshow("Pose Detection", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.stop()

    def display_results(self, image):
        if self.output_segmentation_masks and self.latest_result.segmentation_masks is not None:
            segmentation_mask = self.latest_result.segmentation_masks[0].numpy_view()
            mask_color = (100, 100, 0)  # cyan
            overlay_alpha = 0.5
            mask_image = np.zeros(image.shape, dtype=np.uint8)
            mask_image[:] = mask_color
            condition = np.stack((segmentation_mask,) * 3, axis=-1) > 0.1
            visualized_mask = np.where(condition, mask_image, image)
            image = cv2.addWeighted(image, overlay_alpha, visualized_mask, overlay_alpha, 0)

        for pose_landmarks in self.latest_result.pose_landmarks:
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            mp.solutions.drawing_utils.draw_landmarks(
                image,
                pose_landmarks_proto,
                mp.solutions.pose.POSE_CONNECTIONS)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', help='Name of the pose landmarker model bundle.', required=False, default='pose_landmarker.task')
    parser.add_argument('--numPoses', help='Max number of poses that can be detected by the landmarker.', required=False, default=1)
    parser.add_argument('--minPoseDetectionConfidence', help='The minimum confidence score for pose detection to be considered successful.', required=False, default=0.5)
    parser.add_argument('--minPosePresenceConfidence', help='The minimum confidence score of pose presence score in the pose landmark detection.', required=False, default=0.5)
    parser.add_argument('--minTrackingConfidence', help='The minimum confidence score for the pose tracking to be considered successful.', required=False, default=0.5)
    parser.add_argument('--outputSegmentationMasks', help='Set this if you would also like to visualize the segmentation mask.', required=False, action='store_true')
    parser.add_argument('--cameraId', help='Id of camera.', required=False, default=0)
    parser.add_argument('--frameWidth', help='Width of frame to capture from camera.', required=False, default=1280)
    parser.add_argument('--frameHeight', help='Height of frame to capture from camera.', required=False, default=960)
    args = parser.parse_args()

    detector = PoseDetector(args.model, int(args.numPoses), args.minPoseDetectionConfidence,
                            args.minPosePresenceConfidence, args.minTrackingConfidence,
                            args.outputSegmentationMasks, int(args.cameraId), args.frameWidth, args.frameHeight)
    detector.start()

    try:
        while True:
            detector.run_once()
    except KeyboardInterrupt:
        pass
    finally:
        detector.stop()

if __name__ == '__main__':
    main()
