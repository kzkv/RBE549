# Tom Kazakov
# RBE 549 Week 13 Assignment: Gesture-driven reaction effects

import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from reactions_aux import draw_debug
from reactions_effects import EffectManager, pick_origin

DEBUG = False

CAMERA_INDEX = 0
MODEL_PATH = "models/gesture_recognizer.task"
NUM_HANDS = 2
WINDOW_NAME = "Reactions"

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


def build_recognizer():
    options = vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=NUM_HANDS,
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.GestureRecognizer.create_from_options(options)


# Returns the unique detected reaction class only if all hands agree. Mixed input
# (e.g. one Thumb_Up + one Victory) yields None so ambiguous gestures do not fire.
def active_reaction_class(result):
    names = {
        g[0].category_name
        for g in result.gestures
        if g and g[0].category_name in REACTIONS
    }
    return next(iter(names)) if len(names) == 1 else None


# IDLE -> CANDIDATE (counting consecutive matching frames) -> FIRING (effect playing,
# new triggers suppressed). Returns the fired class once on the trigger frame.
class ReactionDetector:
    IDLE = "IDLE"
    CANDIDATE = "CANDIDATE"
    FIRING = "FIRING"

    def __init__(self, debounce_frames=DEBOUNCE_FRAMES, effect_frames=EFFECT_FRAMES):
        self.debounce_frames = debounce_frames
        self.effect_frames = effect_frames
        self.remaining = 0
        self._reset()

    def _reset(self):
        self.state = self.IDLE
        self.cls = None
        self.count = 0

    def update(self, active):
        if self.state == self.FIRING:
            self.remaining -= 1
            if self.remaining > 0:
                return None
            self._reset()
        if active is None:
            self._reset()
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
    manager = EffectManager()
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
            fired = detector.update(active_reaction_class(result))
            if fired:
                manager.spawn(REACTIONS[fired], pick_origin(result, fired, frame.shape))
            manager.update_and_draw(frame)
            if DEBUG:
                draw_debug(frame, result, detector, REACTIONS)
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
