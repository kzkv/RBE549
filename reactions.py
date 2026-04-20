# Tom Kazakov
# RBE 549 Week 13 Assignment: Gesture-driven reaction effects

import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

CAMERA_INDEX = 0
MODEL_PATH = "models/gesture_recognizer.task"
NUM_HANDS = 2
WINDOW_NAME = "Reactions"

LANDMARK_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 255, 255)
LANDMARK_RADIUS = 3
CONNECTION_THICKNESS = 2

LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 0.6
LABEL_THICKNESS = 1
LABEL_TEXT_COLOR = (255, 255, 255)
LABEL_BG_COLOR = (0, 0, 0)
LABEL_PAD = 4
LABEL_OFFSET_Y = 30

PAIR_LABEL_SCALE = 0.8
PAIR_LABEL_THICKNESS = 2
PAIR_LABEL_ORIGIN = (10, 30)

HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
)


def build_recognizer():
    options = vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=NUM_HANDS,
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.GestureRecognizer.create_from_options(options)


def hand_label(result, i):
    handedness = result.handedness[i][0].category_name if result.handedness[i] else "?"
    gestures = result.gestures[i]
    if gestures:
        top = gestures[0]
        return handedness, top.category_name, top.score
    return handedness, "None", 0.0


def pair_key(result):
    if len(result.hand_landmarks) != 2:
        return None
    names = []
    for i in range(2):
        gestures = result.gestures[i]
        names.append(gestures[0].category_name if gestures else "None")
    return frozenset(names)


def draw_text_box(frame, text, origin, scale, thickness):
    x, y = origin
    (tw, th), baseline = cv2.getTextSize(text, LABEL_FONT, scale, thickness)
    cv2.rectangle(
        frame,
        (x - LABEL_PAD, y - th - LABEL_PAD),
        (x + tw + LABEL_PAD, y + baseline + LABEL_PAD),
        LABEL_BG_COLOR,
        -1,
    )
    cv2.putText(
        frame, text, (x, y), LABEL_FONT, scale, LABEL_TEXT_COLOR, thickness, cv2.LINE_AA
    )


def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for i, j in HAND_CONNECTIONS:
        cv2.line(frame, points[i], points[j], CONNECTION_COLOR, CONNECTION_THICKNESS)
    for x, y in points:
        cv2.circle(frame, (x, y), LANDMARK_RADIUS, LANDMARK_COLOR, -1)


def draw_hand_label(frame, landmarks, handedness, gesture, score):
    h, w = frame.shape[:2]
    wrist = landmarks[0]
    origin = (int(wrist.x * w), int(wrist.y * h) + LABEL_OFFSET_Y)
    text = f"{handedness} {gesture} {score:.2f}"
    draw_text_box(frame, text, origin, LABEL_SCALE, LABEL_THICKNESS)


def draw_pair_label(frame, key):
    text = "Pair: {" + ", ".join(sorted(key)) + "}"
    draw_text_box(
        frame, text, PAIR_LABEL_ORIGIN, PAIR_LABEL_SCALE, PAIR_LABEL_THICKNESS
    )


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {CAMERA_INDEX}")
    recognizer = build_recognizer()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.perf_counter() * 1000)
            result = recognizer.recognize_for_video(mp_image, timestamp_ms)
            for i, landmarks in enumerate(result.hand_landmarks):
                draw_hand(frame, landmarks)
                handedness, gesture, score = hand_label(result, i)
                draw_hand_label(frame, landmarks, handedness, gesture, score)
            pair = pair_key(result)
            if pair is not None:
                draw_pair_label(frame, pair)
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cap.release()
        recognizer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
