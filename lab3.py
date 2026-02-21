# Tom Kazakov
# RBE 549 Lab 3: Image Gradient and Canny Edge Detection

import cv2
import numpy as np

SOBEL_KSIZE_MIN = 1
SOBEL_KSIZE_MAX = 7

SOBEL_KERNEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)

SOBEL_KERNEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

LAPLACIAN_KERNEL = np.array([[2, 0, 2], [0, -8, 0], [2, 0, 2]], dtype=np.float64)

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
        "laplacian_enabled": False,
        "quad_view": False,
    }


def setup_trackbars(window_name, state):
    """Create Lab 3 trackbars."""

    def on_sobel_ksize(v):
        v = max(v, SOBEL_KSIZE_MIN)
        v = v + 1 if v % 2 == 0 else v
        v = min(v, SOBEL_KSIZE_MAX)
        state["sobel_ksize"] = v

    cv2.createTrackbar("Sobel ksize: ", window_name, 3, SOBEL_KSIZE_MAX, on_sobel_ksize)

    def on_canny_thresh1(v):
        state["canny_thresh1"] = max(v, CANNY_THRESH_MIN)

    def on_canny_thresh2(v):
        state["canny_thresh2"] = max(v, CANNY_THRESH_MIN)

    cv2.createTrackbar(
        "Canny T1: ",
        window_name,
        CANNY_THRESH1_DEFAULT,
        CANNY_THRESH_MAX,
        on_canny_thresh1,
    )
    cv2.createTrackbar(
        "Canny T2: ",
        window_name,
        CANNY_THRESH2_DEFAULT,
        CANNY_THRESH_MAX,
        on_canny_thresh2,
    )


def custom_filter(gray, kernel):
    """Apply a convolution kernel via filter2D instead of cv2.Sobel()/cv2.Laplacian()."""
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    filtered = cv2.filter2D(blurred, cv2.CV_64F, kernel)
    return np.uint8(np.absolute(filtered))


QUAD_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
QUAD_LABEL_SCALE = 0.8
QUAD_LABEL_THICKNESS = 2


def _draw_label(img, text):
    """Draw a label centered at the top of a quad panel."""
    (tw, th), _ = cv2.getTextSize(text, QUAD_LABEL_FONT, QUAD_LABEL_SCALE, QUAD_LABEL_THICKNESS)
    x = (img.shape[1] - tw) // 2
    y = th + 10
    cv2.putText(img, text, (x, y), QUAD_LABEL_FONT, QUAD_LABEL_SCALE,
                (0, 0, 0), QUAD_LABEL_THICKNESS + 2)
    cv2.putText(img, text, (x, y), QUAD_LABEL_FONT, QUAD_LABEL_SCALE,
                (255, 255, 255), QUAD_LABEL_THICKNESS)


def build_quad_view(img):
    """Build a 2x2 grid: Original, Laplacian, Sobel X, Sobel Y using custom kernels."""
    h, w = img.shape[:2]
    half_h, half_w = h // 2, w // 2
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    panels = [
        ("Original", cv2.resize(img, (half_w, half_h))),
        ("Laplacian", cv2.resize(cv2.cvtColor(custom_filter(gray, LAPLACIAN_KERNEL), cv2.COLOR_GRAY2BGR), (half_w, half_h))),
        ("Sobel X", cv2.resize(cv2.cvtColor(custom_filter(gray, SOBEL_KERNEL_X), cv2.COLOR_GRAY2BGR), (half_w, half_h))),
        ("Sobel Y", cv2.resize(cv2.cvtColor(custom_filter(gray, SOBEL_KERNEL_Y), cv2.COLOR_GRAY2BGR), (half_w, half_h))),
    ]

    for label, panel in panels:
        _draw_label(panel, label)

    top = np.hstack([panels[0][1], panels[1][1]])
    bottom = np.hstack([panels[2][1], panels[3][1]])
    return np.vstack([top, bottom])


def apply_effects(img, state):
    """Apply Lab 3 gradient/edge effects."""
    if state["quad_view"]:
        return build_quad_view(img)
    mode = state["gradient_mode"]
    if mode in ("sobel_x", "sobel_y"):
        ksize = state["sobel_ksize"]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dx, dy = (1, 0) if mode == "sobel_x" else (0, 1)
        sobel = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=ksize)
        sobel = np.uint8(np.absolute(sobel))
        return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
    if state["laplacian_enabled"]:
        ksize = state["sobel_ksize"]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
        lap = np.uint8(np.absolute(lap))
        return cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)
    if state["canny_enabled"]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, state["canny_thresh1"], state["canny_thresh2"])
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return img


def handle_key(key, state):
    """Handle Lab 3 key presses. Returns True if key was handled."""
    if key == ord("4"):
        state["quad_view"] = not state["quad_view"]
        if state["quad_view"]:
            state["gradient_mode"] = None
            state["gradient_pending"] = False
            state["canny_enabled"] = False
            state["laplacian_enabled"] = False
            print("Quad view: enabled")
        else:
            print("Quad view: disabled")
        return True

    if key == ord("l"):
        state["laplacian_enabled"] = not state["laplacian_enabled"]
        if state["laplacian_enabled"]:
            state["gradient_mode"] = None
            state["gradient_pending"] = False
            state["canny_enabled"] = False
            state["custom_mode"] = None
            print("Laplacian: enabled")
        else:
            print("Laplacian: disabled")
        return True

    if key == ord("d"):
        state["canny_enabled"] = not state["canny_enabled"]
        if state["canny_enabled"]:
            state["gradient_mode"] = None
            state["gradient_pending"] = False
            print(
                f"Canny: enabled (T1={state['canny_thresh1']}, T2={state['canny_thresh2']})"
            )
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
