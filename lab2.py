# Tom Kazakov
# RBE 549 Lab 2: Color spaces, transformations, thresholding, and filtering

import cv2
import numpy as np
import time
from pathlib import Path

BORDER_SIZE = 18
BORDER_COLOR = (0, 0, 255)

LOGO_PATH = Path("OpenCV.png")
LOGO_HEIGHT = 80
LOGO_ALPHA = 0.6

COLOR_SAMPLE_KERNEL = 11
COLOR_HUE_TOLERANCE = 10
COLOR_SAT_TOLERANCE = 50
COLOR_VAL_TOLERANCE = 50

ROTATION_STEP = 10

FLASH_DURATION_MS = 100

THRESHOLD_TYPES = [
    (cv2.THRESH_BINARY, "BINARY"),
    (cv2.THRESH_BINARY_INV, "BINARY_INV"),
    (cv2.THRESH_TRUNC, "TRUNC"),
    (cv2.THRESH_TOZERO, "TOZERO"),
    (cv2.THRESH_TOZERO_INV, "TOZERO_INV"),
]

BLUR_SIGMA_MIN = 5
BLUR_SIGMA_MAX = 30

SHARPEN_AMOUNT_MIN = 5
SHARPEN_AMOUNT_MAX = 30


def init_state():
    """Return Lab 2 state keys with defaults."""
    return {
        "rotation": 0,
        "extract_mode": None,
        "mouse_pos": (0, 0),
        "target_hsv": None,
        "threshold_idx": None,
        "blur_enabled": False,
        "blur_sigma": BLUR_SIGMA_MIN,
        "sharpen_enabled": False,
        "sharpen_amount": SHARPEN_AMOUNT_MIN,
        "flash_start_time": None,
    }


def load_logo():
    """Load and resize logo for blending."""
    logo = cv2.imread(str(LOGO_PATH))
    if logo is None:
        return None, 0, 0
    aspect = logo.shape[1] / logo.shape[0]
    logo = cv2.resize(logo, (int(LOGO_HEIGHT * aspect), LOGO_HEIGHT))
    return logo, logo.shape[0], logo.shape[1]


def setup_trackbars(window_name, state):
    """Create Lab 2 trackbars."""

    def on_blur_sigma(v):
        if v < BLUR_SIGMA_MIN:
            cv2.setTrackbarPos("Blur Sigma: ", window_name, BLUR_SIGMA_MIN)
            state["blur_sigma"] = BLUR_SIGMA_MIN
        else:
            state["blur_sigma"] = v

    def on_sharpen_amount(v):
        if v < SHARPEN_AMOUNT_MIN:
            cv2.setTrackbarPos("Sharpen: ", window_name, SHARPEN_AMOUNT_MIN)
            state["sharpen_amount"] = SHARPEN_AMOUNT_MIN
        else:
            state["sharpen_amount"] = v

    cv2.createTrackbar(
        "Blur Sigma: ", window_name, BLUR_SIGMA_MIN, BLUR_SIGMA_MAX, on_blur_sigma
    )
    cv2.createTrackbar(
        "Sharpen: ",
        window_name,
        SHARPEN_AMOUNT_MIN,
        SHARPEN_AMOUNT_MAX,
        on_sharpen_amount,
    )


def setup_mouse_callback(window_name, state):
    """Set up mouse callback for color sampling."""

    def on_mouse(event, x, y, flags, param):
        state["mouse_pos"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN and state["extract_mode"] == "sampling":
            state["extract_mode"] = "extracting"

    cv2.setMouseCallback(window_name, on_mouse)


def sample_color_at(img, x, y, kernel_size=COLOR_SAMPLE_KERNEL):
    """Sample average HSV color from a kernel region around (x, y)."""
    h, w = img.shape[:2]
    half = kernel_size // 2
    x1, y1 = max(0, x - half), max(0, y - half)
    x2, y2 = min(w, x + half + 1), min(h, y + half + 1)

    region = img[y1:y2, x1:x2]
    hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    return hsv_region.mean(axis=(0, 1)).astype(int)


def get_color_mask(img, target_hsv):
    """Create a mask for pixels matching the target HSV color within tolerances."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array(
        [
            max(0, target_hsv[0] - COLOR_HUE_TOLERANCE),
            max(0, target_hsv[1] - COLOR_SAT_TOLERANCE),
            max(0, target_hsv[2] - COLOR_VAL_TOLERANCE),
        ]
    )
    upper = np.array(
        [
            min(179, target_hsv[0] + COLOR_HUE_TOLERANCE),
            min(255, target_hsv[1] + COLOR_SAT_TOLERANCE),
            min(255, target_hsv[2] + COLOR_VAL_TOLERANCE),
        ]
    )
    return cv2.inRange(hsv, lower, upper)


def apply_rotation(img, rotation):
    """Apply rotation to image."""
    if rotation == 0:
        return img
    h, w = img.shape[:2]
    center = ((w - 1) / 2.0, (h - 1) / 2.0)
    rot_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
    return cv2.warpAffine(img, rot_matrix, (w, h))


def apply_effects(img, state):
    """Apply rotation, color extraction, thresholding, and blur/sharpen effects."""
    result = img.copy()

    result = apply_rotation(result, state["rotation"])

    if state["extract_mode"] == "extracting" and state["target_hsv"] is not None:
        mask = get_color_mask(result, state["target_hsv"])
        result = cv2.bitwise_and(result, result, mask=mask)

    if state["threshold_idx"] is not None:
        thresh_type, _ = THRESHOLD_TYPES[state["threshold_idx"]]
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, threshed = cv2.threshold(gray, 127, 255, thresh_type)
        result = cv2.cvtColor(threshed, cv2.COLOR_GRAY2BGR)

    if state["blur_enabled"]:
        sigma = state["blur_sigma"]
        result = cv2.GaussianBlur(result, (0, 0), sigma, sigma)
    elif state["sharpen_enabled"]:
        amount = state["sharpen_amount"] / 10.0
        blurred = cv2.GaussianBlur(result, (0, 0), 3)
        result = cv2.addWeighted(result, 1 + amount, blurred, -amount, 0)

    return result


def update_color_sample(zoomed, state):
    """Update color sample in sampling mode from rotated image."""
    if state["extract_mode"] != "sampling":
        return

    sample_img = apply_rotation(zoomed, state["rotation"])
    mx, my = state["mouse_pos"]
    mx = max(0, min(mx - BORDER_SIZE, sample_img.shape[1] - 1))
    my = max(0, min(my - BORDER_SIZE, sample_img.shape[0] - 1))
    state["target_hsv"] = sample_color_at(sample_img, mx, my)


def apply_border(img):
    """Apply red border around image."""
    return cv2.copyMakeBorder(
        img,
        BORDER_SIZE,
        BORDER_SIZE,
        BORDER_SIZE,
        BORDER_SIZE,
        cv2.BORDER_CONSTANT,
        value=BORDER_COLOR,
    )


def copy_timestamp_roi(display, roi_bounds):
    """Copy timestamp ROI to the top right corner. Returns roi_h for overlay positioning."""
    roi_y1, roi_y2, roi_x1, roi_x2 = roi_bounds
    timestamp_roi = display[roi_y1:roi_y2, roi_x1:roi_x2].copy()
    roi_h, roi_w = timestamp_roi.shape[:2]
    display[0:roi_h, display.shape[1] - roi_w : display.shape[1]] = timestamp_roi
    return roi_h


def draw_overlays(display, state, logo, logo_h, logo_w, roi_h, is_recording):
    """Draw logo, recording indicator, and sampling preview."""
    if logo is not None:
        roi = display[0:logo_h, 0:logo_w]
        blended = cv2.addWeighted(roi, 1 - LOGO_ALPHA, logo, LOGO_ALPHA, 0)
        display[0:logo_h, 0:logo_w] = blended

    if is_recording and int(time.time()) % 3:
        cv2.circle(display, (display.shape[1] - 20, roi_h + 15), 8, (0, 0, 255), -1)

    if state["extract_mode"] == "sampling" and state["target_hsv"] is not None:
        mx, my = state["mouse_pos"]
        mx = max(0, min(mx - BORDER_SIZE, display.shape[1] - 1))
        my = max(0, min(my - BORDER_SIZE, display.shape[0] - 1))
        half = COLOR_SAMPLE_KERNEL // 2
        cv2.rectangle(
            display, (mx - half, my - half), (mx + half, my + half), (0, 255, 0), 2
        )
        bgr_color = cv2.cvtColor(
            np.array([[state["target_hsv"]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
        )[0, 0]
        cv2.rectangle(
            display,
            (mx + half + 5, my - half),
            (mx + half + 35, my + half),
            bgr_color.tolist(),
            -1,
        )
        cv2.rectangle(
            display,
            (mx + half + 5, my - half),
            (mx + half + 35, my + half),
            (255, 255, 255),
            1,
        )


def apply_flash(display, state):
    """Apply flash effect if active. Returns True if flash is active."""
    if state["flash_start_time"] is None:
        return False

    elapsed_ms = (time.time() - state["flash_start_time"]) * 1000
    if elapsed_ms < FLASH_DURATION_MS:
        display[:, :] = (255, 255, 255)
        return True
    else:
        state["flash_start_time"] = None
        return False


def trigger_flash(state):
    """Start flash effect."""
    state["flash_start_time"] = time.time()


def handle_key(key, state):
    """Handle Lab 2 key presses. Returns True if key was handled."""
    if key == ord("e"):
        if state["extract_mode"] is None:
            state["extract_mode"] = "sampling"
            print("Color extraction: sampling mode - click to select color")
        else:
            state["extract_mode"] = None
            state["target_hsv"] = None
            print("Color extraction: disabled")
        return True

    elif key == ord("r"):
        state["rotation"] = (state["rotation"] - ROTATION_STEP) % 360
        print(f"Rotation: {state['rotation']}° (CW)")
        return True

    elif key == ord("R"):
        state["rotation"] = (state["rotation"] + ROTATION_STEP) % 360
        print(f"Rotation: {state['rotation']}° (CCW)")
        return True

    elif key == ord("t"):
        if state["threshold_idx"] is None:
            state["threshold_idx"] = 0
        else:
            state["threshold_idx"] = (state["threshold_idx"] + 1) % len(THRESHOLD_TYPES)
        _, name = THRESHOLD_TYPES[state["threshold_idx"]]
        print(f"Threshold: {name}")
        return True

    elif key == ord("T"):
        state["threshold_idx"] = None
        print("Threshold: disabled")
        return True

    elif key == ord("b"):
        state["blur_enabled"] = not state["blur_enabled"]
        if state["blur_enabled"]:
            state["sharpen_enabled"] = False
            print(f"Blur: enabled (sigma={state['blur_sigma']})")
        else:
            print("Blur: disabled")
        return True

    elif key == ord("s"):
        state["sharpen_enabled"] = not state["sharpen_enabled"]
        if state["sharpen_enabled"]:
            state["blur_enabled"] = False
            print(f"Sharpen: enabled (amount={state['sharpen_amount'] / 10.0:.1f})")
        else:
            print("Sharpen: disabled")
        return True

    return False
