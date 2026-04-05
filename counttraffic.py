# Tom Kazakov
# RBE 549 Lab 11: Traffic Monitoring

import cv2
import numpy as np

VIDEO_PATH = "TrafficVideo.mp4"
WINDOW_NAME = "Traffic Monitor"
CROSSWALK_ZONE = np.array([(623, 853), (1297, 606), (1681, 629), (1432, 966)])
CROSSWALK_COLOR = (0, 255, 255)
CROSSWALK_ALPHA = 0.3


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1

        overlay = frame.copy()
        cv2.fillPoly(overlay, [CROSSWALK_ZONE], CROSSWALK_COLOR)
        cv2.addWeighted(overlay, CROSSWALK_ALPHA, frame, 1 - CROSSWALK_ALPHA, 0, frame)
        cv2.polylines(frame, [CROSSWALK_ZONE], True, CROSSWALK_COLOR, 2)

        label = f"{frame_number}/{total_frames}"
        cv2.putText(
            frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
