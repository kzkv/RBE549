# Tom Kazakov
# RBE 549 Lab 11: Traffic Monitoring

import cv2
import numpy as np
from collections import defaultdict
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
START_FRAME = 0
CROSSWALK_ZONE_F32 = None


def is_inside_zone(point):
    """Test if a point is inside the crosswalk polygon."""
    global CROSSWALK_ZONE_F32
    if CROSSWALK_ZONE_F32 is None:
        CROSSWALK_ZONE_F32 = CROSSWALK_ZONE.astype(np.float32)
    return cv2.pointPolygonTest(CROSSWALK_ZONE_F32, point, False) >= 0


def bottom_center(xyxy):
    """Return the bottom-center point of an xyxy bounding box."""
    x1, _, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, float(y2))


EDGE_DIRECTIONS = {
    "through traffic": [
        (CROSSWALK_ZONE[0], CROSSWALK_ZONE[1]),
        (CROSSWALK_ZONE[2], CROSSWALK_ZONE[3]),
    ],
    "pedestrian crossing": [
        (CROSSWALK_ZONE[1], CROSSWALK_ZONE[2]),
        (CROSSWALK_ZONE[3], CROSSWALK_ZONE[0]),
    ],
}


def segments_intersect(p1, p2, p3, p4):
    """Test if line segment p1-p2 intersects segment p3-p4."""
    d1 = (p4[0] - p3[0]) * (p1[1] - p3[1]) - (p4[1] - p3[1]) * (p1[0] - p3[0])
    d2 = (p4[0] - p3[0]) * (p2[1] - p3[1]) - (p4[1] - p3[1]) * (p2[0] - p3[0])
    d3 = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    d4 = (p2[0] - p1[0]) * (p4[1] - p1[1]) - (p2[1] - p1[1]) * (p4[0] - p1[0])
    return d1 * d2 < 0 and d3 * d4 < 0


def classify_direction(p1, p2):
    """Classify crossing direction based on which polygon edge the movement crosses."""
    for direction, edges in EDGE_DIRECTIONS.items():
        for a, b in edges:
            if segments_intersect(p1, p2, a, b):
                return direction
    return "through traffic"


def update_crossings(
    detections, track_inside, track_entry, track_last, counted, crossing_counts
):
    """Track zone entry/exit and count crossings."""
    for track_id, cls_id, xyxy in detections:
        bc = bottom_center(xyxy)
        inside = is_inside_zone(bc)
        was_inside = track_inside.get(track_id)

        if was_inside is None and inside:
            track_entry[track_id] = bc
        elif was_inside is None and not inside:
            pass
        elif inside and not was_inside:
            track_entry[track_id] = bc
        elif not inside and was_inside and track_id in track_entry:
            if track_id not in counted:
                last_inside = track_last.get(track_id, track_entry[track_id])
                direction = classify_direction(last_inside, bc)
                counted.add(track_id)
                class_name = TARGET_CLASSES[cls_id]
                crossing_counts[(class_name, direction)] += 1

        if inside:
            track_last[track_id] = bc
        track_inside[track_id] = inside


def put_text(frame, text, x, y, scale=0.5, color=(255, 255, 255)):
    """Draw text on the frame."""
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def draw_hud(frame, crossing_counts):
    """Draw crossing counts on the top-left of the frame."""
    y = 60
    for direction in EDGE_DIRECTIONS:
        entries = {
            k: v for k, v in crossing_counts.items() if k[1] == direction and v > 0
        }
        if not entries:
            continue
        put_text(frame, f"{direction}:", 10, y, scale=0.6)
        y += 25
        for (class_name, _), count in sorted(entries.items()):
            put_text(frame, f"  {class_name}: {count}", 10, y)
            y += 22
        y += 10


def filter_target_boxes(boxes):
    """Return list of (track_id, cls_id, xyxy) for tracked target objects above confidence."""
    result = []
    for box in boxes:
        cls_id = int(box.cls)
        if cls_id not in TARGET_CLASSES:
            continue
        if float(box.conf) < CONFIDENCE_THRESHOLD:
            continue
        if box.id is None:
            continue
        track_id = int(box.id)
        xyxy = tuple(map(int, box.xyxy[0]))
        result.append((track_id, cls_id, xyxy))
    return result


def draw_detections(frame, detections):
    """Draw bounding boxes with track IDs."""
    for track_id, cls_id, xyxy in detections:
        x1, y1, x2, y2 = xyxy
        color = BOX_COLORS[cls_id]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{TARGET_CLASSES[cls_id]} #{track_id}"
        put_text(frame, label, x1, y1 - 8, color=color)


def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if START_FRAME > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    frame_number = START_FRAME

    track_inside = {}
    track_entry = {}
    track_last = {}
    counted = set()
    crossing_counts = defaultdict(int)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1

        results = model.track(frame, persist=True, verbose=False)
        boxes = results[0].boxes
        detections = filter_target_boxes(boxes)

        update_crossings(
            detections, track_inside, track_entry, track_last, counted, crossing_counts
        )
        draw_detections(frame, detections)

        overlay = frame.copy()
        cv2.fillPoly(overlay, [CROSSWALK_ZONE], CROSSWALK_COLOR)
        cv2.addWeighted(overlay, CROSSWALK_ALPHA, frame, 1 - CROSSWALK_ALPHA, 0, frame)
        cv2.polylines(frame, [CROSSWALK_ZONE], True, CROSSWALK_COLOR, 2)

        draw_hud(frame, crossing_counts)

        put_text(
            frame,
            f"{frame_number}/{total_frames}",
            10,
            30,
            scale=0.7,
            color=(0, 255, 0),
        )

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
