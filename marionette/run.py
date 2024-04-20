from pose_detect import PoseDetector

def main():
    detector = PoseDetector(
        model='pose_landmarker_lite.task',
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
        camera_id=0,
        width=1280,
        height=960)
    detector.start()

    try:
        while True:
            detector.run_once()
            result = detector.get_latest_result()
            if result:
                print("Detected pose:", result)
            # Additional processing can be done here
    except KeyboardInterrupt:
        pass
    finally:
        detector.stop()

if __name__ == '__main__':
    main()
