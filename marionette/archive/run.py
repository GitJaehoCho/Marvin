# run_pose_detection.py
from pose_landmarker import PoseLandmarker

def run_detection():
    # Configuration as before
    landmarker = PoseLandmarker(
        model='pose_landmarker_lite.task',
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=True,
        camera_id=0,
        width=1280,
        height=720
    )

    try:
        # Start the detection process
        landmarker.run_detection()
    finally:
        # Properly release all resources
        landmarker.close()
    
    # Retrieve and process the detection results
    results = landmarker.get_detection_results()
    if results:
        print("Detection results retrieved successfully.")
        # Further processing of results can be done here
    else:
        print("No detection results available.")

if __name__ == '__main__':
    run_detection()
