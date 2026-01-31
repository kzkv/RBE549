# Tom Kazakov
# RBE 549 Lab 3: Image Gradient and Canny Edge Detection

import cv2
import numpy as np

SOBEL_KSIZE_MIN = 1
SOBEL_KSIZE_MAX = 7

SOBEL_KERNEL_X = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float64)

SOBEL_KERNEL_Y = np.array([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=np.float64)

CANNY_THRESH_MIN = 1
CANNY_THRESH_MAX = 5000
CANNY_THRESH1_DEFAULT = 100
CANNY_THRESH2_DEFAULT = 200


def init_state():
    """Return Lab 3 state keys with defaults."""
    return {
        "gradient_pending": False,
        "gradient_mode": None,
        "sobel_ksize": 3,
        "canny_enabled": False,
        "canny_thresh1": CANNY_THRESH1_DEFAULT,
        "canny_thresh2": CANNY_THRESH2_DEFAULT,
        "custom_mode": None,
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

    def on_canny_thresh1(v):
        if v < CANNY_THRESH_MIN:
            cv2.setTrackbarPos("Canny T1: ", window_name, CANNY_THRESH_MIN)
            v = CANNY_THRESH_MIN
        state["canny_thresh1"] = v

    def on_canny_thresh2(v):
        if v < CANNY_THRESH_MIN:
            cv2.setTrackbarPos("Canny T2: ", window_name, CANNY_THRESH_MIN)
            v = CANNY_THRESH_MIN
        state["canny_thresh2"] = v

    cv2.createTrackbar(
        "Canny T1: ", window_name, CANNY_THRESH1_DEFAULT, CANNY_THRESH_MAX, on_canny_thresh1
    )
    cv2.createTrackbar(
        "Canny T2: ", window_name, CANNY_THRESH2_DEFAULT, CANNY_THRESH_MAX, on_canny_thresh2
    )


def custom_filter(gray, kernel):
    """Apply a convolution kernel via filter2D instead of cv2.Sobel()/cv2.Laplacian()."""
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    filtered = cv2.filter2D(blurred, cv2.CV_64F, kernel)
    return np.uint8(np.absolute(filtered))


CUSTOM_MODES = [None, "custom_sobel_x", "custom_sobel_y"]
CUSTOM_KERNELS = {
    "custom_sobel_x": SOBEL_KERNEL_X,
    "custom_sobel_y": SOBEL_KERNEL_Y,
}


def apply_effects(img, state):
    """Apply Lab 3 gradient/edge effects."""
    if state["custom_mode"] in CUSTOM_KERNELS:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = custom_filter(gray, CUSTOM_KERNELS[state["custom_mode"]])
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    mode = state["gradient_mode"]
    if mode in ("sobel_x", "sobel_y"):
        ksize = state["sobel_ksize"]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dx, dy = (1, 0) if mode == "sobel_x" else (0, 1)
        sobel = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=ksize)
        sobel = np.uint8(np.absolute(sobel))
        return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
    if state["canny_enabled"]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, state["canny_thresh1"], state["canny_thresh2"])
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return img


def handle_key(key, state):
    """Handle Lab 3 key presses. Returns True if key was handled."""
    if key == ord("4"):
        idx = CUSTOM_MODES.index(state["custom_mode"])
        state["custom_mode"] = CUSTOM_MODES[(idx + 1) % len(CUSTOM_MODES)]
        if state["custom_mode"]:
            state["gradient_mode"] = None
            state["gradient_pending"] = False
            state["canny_enabled"] = False
            print(f"Custom: {state['custom_mode']}")
        else:
            print("Custom: disabled")
        return True

    if key == ord("d"):
        state["canny_enabled"] = not state["canny_enabled"]
        if state["canny_enabled"]:
            state["gradient_mode"] = None
            state["gradient_pending"] = False
            print(f"Canny: enabled (T1={state['canny_thresh1']}, T2={state['canny_thresh2']})")
        else:
            print("Canny: disabled")
        return True

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
            state["canny_enabled"] = False
            print("Gradient: Sobel X enabled")
            return True
        elif key == ord("y"):
            state["gradient_mode"] = "sobel_y"
            state["gradient_pending"] = False
            state["canny_enabled"] = False
            print("Gradient: Sobel Y enabled")
            return True
        elif state["gradient_pending"] and key != -1:
            state["gradient_pending"] = False
            print("Gradient: cancelled")
            return True

    return False
