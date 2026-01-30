# Tom Kazakov
# RBE 549 Lab 3: Image Gradient and Canny Edge Detection

import cv2
import numpy as np

SOBEL_KSIZE_MIN = 1
SOBEL_KSIZE_MAX = 7


def init_state():
    """Return Lab 3 state keys with defaults."""
    return {
        "gradient_pending": False,
        "gradient_mode": None,
        "sobel_ksize": 3,
    }


def setup_trackbars(window_name, state):
    """Create Lab 3 trackbars."""

    def on_sobel_ksize(v):
        if v < SOBEL_KSIZE_MIN:
            v = SOBEL_KSIZE_MIN
        elif v % 2 == 0:
            v = v + 1
        if v > SOBEL_KSIZE_MAX:
            v = SOBEL_KSIZE_MAX
        cv2.setTrackbarPos("Sobel ksize: ", window_name, v)
        state["sobel_ksize"] = v

    cv2.createTrackbar("Sobel ksize: ", window_name, 3, SOBEL_KSIZE_MAX, on_sobel_ksize)


def apply_effects(img, state):
    """Apply Lab 3 gradient/edge effects."""
    mode = state["gradient_mode"]
    if mode in ("sobel_x", "sobel_y"):
        ksize = state["sobel_ksize"]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dx, dy = (1, 0) if mode == "sobel_x" else (0, 1)
        sobel = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=ksize)
        sobel = np.uint8(np.absolute(sobel))
        return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
    return img


def handle_key(key, state):
    """Handle Lab 3 key presses. Returns True if key was handled."""
    if key == ord("g"):
        if state["gradient_mode"] is not None:
            state["gradient_mode"] = None
            state["gradient_pending"] = False
            print("Gradient: disabled")
        else:
            state["gradient_pending"] = True
            print("Gradient: press x for Sobel X, y for Sobel Y")
        return True

    if state["gradient_pending"] or state["gradient_mode"] in ("sobel_x", "sobel_y"):
        if key == ord("x"):
            state["gradient_mode"] = "sobel_x"
            state["gradient_pending"] = False
            print("Gradient: Sobel X enabled")
            return True
        elif key == ord("y"):
            state["gradient_mode"] = "sobel_y"
            state["gradient_pending"] = False
            print("Gradient: Sobel Y enabled")
            return True
        elif state["gradient_pending"] and key != -1:
            state["gradient_pending"] = False
            print("Gradient: cancelled")
            return True

    return False
