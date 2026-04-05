# Tom Kazakov
# RBE 549 Lab 11: Traffic Monitoring

import time
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

VIDEO_PATH = "TrafficVideo.mp4"
MODEL_PATH = "yolo11n.pt"
WINDOW_NAME = "Traffic Monitor"
OUTPUT_PATH = "week11_outcome.mp4"
RECORD = True
START_FRAME = 0
CONFIDENCE_THRESHOLD = 0.4
EXIT_DEBOUNCE_FRAMES = 5

TARGET_CLASSES = {0: "human", 1: "bike", 2: "car"}
BOX_COLORS = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 0, 255)}

CROSSWALK_ZONE = np.array([(623, 853), (1297, 606), (1681, 629), (1432, 966)])
CROSSWALK_ZONE_F32 = CROSSWALK_ZONE.astype(np.float32)
CROSSWALK_COLOR = (0, 255, 255)
CROSSWALK_ALPHA = 0.3

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


def bottom_center(xyxy):
    """Return the bottom-center point of an xyxy bounding box."""
    x1, _, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, float(y2))


def is_inside_zone(point):
    """Test if a point is inside the crosswalk polygon."""
    return cv2.pointPolygonTest(CROSSWALK_ZONE_F32, point, False) >= 0


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


def put_text(frame, text, x, y, scale=0.5, color=(255, 255, 255)):
    """Draw text on the frame."""
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def filter_target_boxes(boxes):
    """Return list of (track_id, cls_id, xyxy) for tracked target objects above confidence."""
    return [
        (int(box.id), int(box.cls), tuple(map(int, box.xyxy[0])))
        for box in boxes
        if box.id is not None
        and int(box.cls) in TARGET_CLASSES
        and float(box.conf) >= CONFIDENCE_THRESHOLD
    ]


class CrossingTracker:
    """Tracks objects entering and exiting the crosswalk zone with debounce."""

    def __init__(self):
        self.inside = {}
        self.entry = {}
        self.last_inside = {}
        self.outside_count = {}
        self.counted = set()
        self.counts = defaultdict(int)

    def update(self, frame_number, detections):
        """Process one frame of detections."""
        for track_id, cls_id, xyxy in detections:
            bc = bottom_center(xyxy)
            currently_inside = is_inside_zone(bc)
            was_inside = self.inside.get(track_id)

            if currently_inside:
                self._handle_inside(frame_number, track_id, cls_id, bc, was_inside)
            else:
                self._handle_outside(frame_number, track_id, cls_id, bc, was_inside)

    def _handle_inside(self, frame_number, track_id, cls_id, bc, was_inside):
        self.outside_count.pop(track_id, None)
        if not was_inside:
            self.entry[track_id] = bc
            print(
                f"[{frame_number}] #{track_id} {TARGET_CLASSES[cls_id]} ENTERED at {bc}"
            )
        self.last_inside[track_id] = bc
        self.inside[track_id] = True

    def _handle_outside(self, frame_number, track_id, cls_id, bc, was_inside):
        if not was_inside:
            self.inside[track_id] = False
            return
        self.outside_count[track_id] = self.outside_count.get(track_id, 0) + 1
        if self.outside_count[track_id] < EXIT_DEBOUNCE_FRAMES:
            return
        print(f"[{frame_number}] #{track_id} {TARGET_CLASSES[cls_id]} EXITED at {bc}")
        self.inside[track_id] = False
        self.outside_count.pop(track_id, None)
        if track_id in self.entry and track_id not in self.counted:
            origin = self.last_inside.get(track_id, self.entry[track_id])
            direction = classify_direction(origin, bc)
            self.counted.add(track_id)
            self.counts[(TARGET_CLASSES[cls_id], direction)] += 1
            print(f"  -> COUNTED as {TARGET_CLASSES[cls_id]} {direction}")


def draw_detections(frame, detections):
    """Draw bounding boxes with track IDs."""
    for track_id, cls_id, xyxy in detections:
        x1, y1, x2, y2 = xyxy
        color = BOX_COLORS[cls_id]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        put_text(
            frame, f"{TARGET_CLASSES[cls_id]} #{track_id}", x1, y1 - 8, color=color
        )


def draw_crosswalk(frame):
    """Draw the crosswalk zone as a semi-transparent overlay."""
    overlay = frame.copy()
    cv2.fillPoly(overlay, [CROSSWALK_ZONE], CROSSWALK_COLOR)
    cv2.addWeighted(overlay, CROSSWALK_ALPHA, frame, 1 - CROSSWALK_ALPHA, 0, frame)
    cv2.polylines(frame, [CROSSWALK_ZONE], True, CROSSWALK_COLOR, 2)


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


def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if START_FRAME > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    frame_number = START_FRAME
    tracker = CrossingTracker()

    writer = None
    if RECORD:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1
        t0 = time.perf_counter()

        results = model.track(frame, persist=True, verbose=False)
        detections = filter_target_boxes(results[0].boxes)

        tracker.update(frame_number, detections)
        draw_detections(frame, detections)
        draw_crosswalk(frame)
        draw_hud(frame, tracker.counts)

        processing_fps = 1.0 / max(time.perf_counter() - t0, 1e-9)
        put_text(
            frame,
            f"{frame_number}/{total_frames} | {processing_fps:.1f} fps",
            10,
            30,
            scale=0.7,
            color=(0, 255, 0),
        )

        if writer:
            writer.write(frame)
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if writer:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
