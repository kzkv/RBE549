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

STATE_ORIGIN = (10, 30)
STATE_SCALE = 0.7
STATE_THICKNESS = 1

BANNER_SCALE = 1.4
BANNER_THICKNESS = 3
BANNER_TEXT_COLOR = (255, 255, 255)
BANNER_BG_COLOR = (32, 32, 32)

DEBOUNCE_FRAMES = 6
EFFECT_FRAMES = 60

REACTIONS = {
    "Thumb_Up": "thumbs_up",
    "Thumb_Down": "thumbs_down",
    "Closed_Fist": "balloons",
    "Pointing_Up": "rain",
    "Victory": "confetti",
    "Open_Palm": "hearts",
}

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


def active_reaction_class(result):
    names = {
        g[0].category_name
        for g in result.gestures
        if g and g[0].category_name in REACTIONS
    }
    return next(iter(names)) if len(names) == 1 else None


def draw_text_box(
    frame, text, origin, scale, thickness, fg=LABEL_TEXT_COLOR, bg=LABEL_BG_COLOR
):
    x, y = origin
    (tw, th), baseline = cv2.getTextSize(text, LABEL_FONT, scale, thickness)
    cv2.rectangle(
        frame,
        (x - LABEL_PAD, y - th - LABEL_PAD),
        (x + tw + LABEL_PAD, y + baseline + LABEL_PAD),
        bg,
        -1,
    )
    cv2.putText(frame, text, (x, y), LABEL_FONT, scale, fg, thickness, cv2.LINE_AA)


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


def draw_state_indicator(frame, detector):
    if detector.state == ReactionDetector.IDLE:
        text = "IDLE"
    elif detector.state == ReactionDetector.CANDIDATE:
        text = f"CANDIDATE {detector.cls} {detector.count}/{detector.debounce_frames}"
    else:
        reaction_id = REACTIONS.get(detector.cls, "?")
        text = f"FIRING {reaction_id} {detector.remaining}/{detector.effect_frames}"
    draw_text_box(frame, text, STATE_ORIGIN, STATE_SCALE, STATE_THICKNESS)


def draw_reaction_banner(frame, reaction_id):
    h, w = frame.shape[:2]
    text = f"REACTION: {reaction_id}"
    (tw, _), _ = cv2.getTextSize(text, LABEL_FONT, BANNER_SCALE, BANNER_THICKNESS)
    origin = ((w - tw) // 2, h // 2)
    draw_text_box(
        frame,
        text,
        origin,
        BANNER_SCALE,
        BANNER_THICKNESS,
        BANNER_TEXT_COLOR,
        BANNER_BG_COLOR,
    )


class ReactionDetector:
    IDLE = "IDLE"
    CANDIDATE = "CANDIDATE"
    FIRING = "FIRING"

    def __init__(self, debounce_frames=DEBOUNCE_FRAMES, effect_frames=EFFECT_FRAMES):
        self.debounce_frames = debounce_frames
        self.effect_frames = effect_frames
        self.state = self.IDLE
        self.cls = None
        self.count = 0
        self.remaining = 0

    def update(self, active):
        if self.state == self.FIRING:
            self.remaining -= 1
            if self.remaining > 0:
                return None
            self.state = self.IDLE
            self.cls = None
            self.count = 0
        if active is None:
            self.state = self.IDLE
            self.cls = None
            self.count = 0
            return None
        if active != self.cls:
            self.state = self.CANDIDATE
            self.cls = active
            self.count = 1
            return None
        self.count += 1
        if self.count >= self.debounce_frames:
            self.state = self.FIRING
            self.remaining = self.effect_frames
            return self.cls
        return None


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {CAMERA_INDEX}")
    recognizer = build_recognizer()
    detector = ReactionDetector()
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
            detector.update(active_reaction_class(result))
            draw_state_indicator(frame, detector)
            if detector.state == ReactionDetector.FIRING:
                draw_reaction_banner(frame, REACTIONS[detector.cls])
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
