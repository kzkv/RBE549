# Tom Kazakov
# RBE 549 Lab 1: Zoom and timestamp features

import cv2
from datetime import datetime

TIMESTAMP_FONT = cv2.FONT_HERSHEY_PLAIN
TIMESTAMP_SCALE = 2.0
TIMESTAMP_COLOR = (0, 165, 255)
TIMESTAMP_THICKNESS = 3
TIMESTAMP_PADDING = 5


def init_state():
    """Return Lab 1 state keys with defaults."""
    return {"zoom": 0}


def setup_trackbars(window_name, state):
    """Create Lab 1 trackbars."""
    cv2.createTrackbar("Zoom, %: ", window_name, 0, 100, lambda v: state.update(zoom=v))


def apply_zoom(img, zoom_pct):
    """Apply digital zoom by cropping and resizing."""
    if zoom_pct == 100:
        return img
    h, w = img.shape[:2]
    crop_h, crop_w = int(h * 100 / zoom_pct), int(w * 100 / zoom_pct)
    y1, x1 = (h - crop_h) // 2, (w - crop_w) // 2
    cropped = img[y1 : y1 + crop_h, x1 : x1 + crop_w]
    return cv2.resize(cropped, (w, h))


def draw_timestamp(img, include_seconds=True):
    """Draw timestamp at bottom right and return its ROI bounds (y1, y2, x1, x2)."""
    h, w = img.shape[:2]
    fmt = "%m %d '%y  %H:%M:%S" if include_seconds else "%m %d '%y  %H:%M"
    text = datetime.now().strftime(fmt)

    (text_w, text_h), baseline = cv2.getTextSize(
        text, TIMESTAMP_FONT, TIMESTAMP_SCALE, TIMESTAMP_THICKNESS
    )

    x = w - text_w - TIMESTAMP_PADDING
    y = h - TIMESTAMP_PADDING - baseline
    cv2.putText(
        img,
        text,
        (x, y),
        TIMESTAMP_FONT,
        TIMESTAMP_SCALE,
        TIMESTAMP_COLOR,
        TIMESTAMP_THICKNESS,
    )

    roi_x1 = x - TIMESTAMP_PADDING
    roi_x2 = w
    roi_y1 = y - text_h - TIMESTAMP_PADDING
    roi_y2 = h
    return roi_y1, roi_y2, roi_x1, roi_x2


def handle_key(key, state, window_name):
    """Handle Lab 1 key presses. Returns True if key was handled."""
    if key in (ord("+"), ord("=")):
        state["zoom"] = min(100, state["zoom"] + 10)
        cv2.setTrackbarPos("Zoom, %: ", window_name, state["zoom"])
        return True
    elif key in (ord("-"), ord("_")):
        state["zoom"] = max(0, state["zoom"] - 10)
        cv2.setTrackbarPos("Zoom, %: ", window_name, state["zoom"])
        return True
    return False
