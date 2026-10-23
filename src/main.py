# src/main.py
import cv2
from drowsiness_detector import DrowsinessDetector

def main():
    detector = DrowsinessDetector(
        ear_thresh=0.25,            # eyes fully closed ~ 0.22
        ear_consec_seconds=7,     # 1.5 sec continuous closure

        head_pitch_thresh_deg=10.0, # practical threshold
        head_consec_seconds=7,    # 2 seconds downward

        mar_thresh=0.55,            # yawn threshold
        mar_consec_seconds=30      # yawn lasts 2 sec
    )

    cap = cv2.VideoCapture(0)  # change index if you have multiple cameras
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set a reasonable frame size (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror horizontally for user-friendly display
            frame = cv2.flip(frame, 1)

            annotated, alarm, info = detector.process_frame(frame)
            cv2.imshow("Drowsiness Detector", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
