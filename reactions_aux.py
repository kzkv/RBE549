# Tom Kazakov
# RBE 549 Week 13 Assignment: Debug overlays for reactions.py

import cv2

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

LANDMARK_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 255, 255)
LANDMARK_RADIUS = 3
CONNECTION_THICKNESS = 2

FONT = cv2.FONT_HERSHEY_SIMPLEX
PAD = 4
LABEL_TEXT = (255, 255, 255)
LABEL_BG = (0, 0, 0)

HAND_LABEL_SCALE = 0.6
HAND_LABEL_THICKNESS = 1

STATE_ORIGIN = (10, 30)
STATE_SCALE = 0.7
STATE_THICKNESS = 1

BANNER_SCALE = 1.4
BANNER_THICKNESS = 3
BANNER_BG = (32, 32, 32)


def _text_box(frame, text, origin, scale, thickness, bg=LABEL_BG):
    x, y = origin
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    cv2.rectangle(
        frame,
        (x - PAD, y - th - PAD),
        (x + tw + PAD, y + baseline + PAD),
        bg,
        -1,
    )
    cv2.putText(frame, text, (x, y), FONT, scale, LABEL_TEXT, thickness, cv2.LINE_AA)


def _draw_hand_debug(frame, result, i):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in result.hand_landmarks[i]]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], CONNECTION_COLOR, CONNECTION_THICKNESS)
    for x, y in points:
        cv2.circle(frame, (x, y), LANDMARK_RADIUS, LANDMARK_COLOR, -1)
    handedness = result.handedness[i][0].category_name if result.handedness[i] else "?"
    gestures = result.gestures[i]
    name, score = (
        (gestures[0].category_name, gestures[0].score) if gestures else ("None", 0.0)
    )
    origin = (points[0][0], points[0][1] + 30)
    _text_box(
        frame,
        f"{handedness} {name} {score:.2f}",
        origin,
        HAND_LABEL_SCALE,
        HAND_LABEL_THICKNESS,
    )


def _draw_state(frame, detector, reactions):
    if detector.state == detector.IDLE:
        text = "IDLE"
    elif detector.state == detector.CANDIDATE:
        text = f"CANDIDATE {detector.cls} {detector.count}/{detector.debounce_frames}"
    else:
        reaction = reactions.get(detector.cls, "?")
        text = f"FIRING {reaction} {detector.remaining}/{detector.effect_frames}"
    _text_box(frame, text, STATE_ORIGIN, STATE_SCALE, STATE_THICKNESS)


def _draw_banner(frame, reaction_id):
    h, w = frame.shape[:2]
    text = f"REACTION: {reaction_id}"
    (tw, _), _ = cv2.getTextSize(text, FONT, BANNER_SCALE, BANNER_THICKNESS)
    origin = ((w - tw) // 2, h // 2)
    _text_box(frame, text, origin, BANNER_SCALE, BANNER_THICKNESS, bg=BANNER_BG)


def draw_debug(frame, result, detector, reactions):
    for i in range(len(result.hand_landmarks)):
        _draw_hand_debug(frame, result, i)
    _draw_state(frame, detector, reactions)
    if detector.state == detector.FIRING:
        _draw_banner(frame, reactions[detector.cls])
