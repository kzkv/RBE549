# Tom Kazakov
# RBE 549 Lab 11: Traffic Monitoring

import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = "TrafficVideo.mp4"
MODEL_PATH = "yolo11n.pt"
WINDOW_NAME = "Traffic Monitor"
CROSSWALK_ZONE = np.array([(623, 853), (1297, 606), (1681, 629), (1432, 966)])
CROSSWALK_COLOR = (0, 255, 255)
CROSSWALK_ALPHA = 0.3
CONFIDENCE_THRESHOLD = 0.4
TARGET_CLASSES = {0: "human", 1: "bike", 2: "car"}
BOX_COLORS = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 0, 255)}
START_FRAME = 500


def draw_detections(frame, results):
    """Draw bounding boxes and labels for detected target objects."""
    for box in results[0].boxes:
        cls_id = int(box.cls)
        if cls_id not in TARGET_CLASSES:
            continue
        if float(box.conf) < CONFIDENCE_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = BOX_COLORS[cls_id]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{TARGET_CLASSES[cls_id]} {float(box.conf):.2f}"
        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if START_FRAME > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    frame_number = START_FRAME

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1

        results = model(frame, verbose=False)
        draw_detections(frame, results)

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
