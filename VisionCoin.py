# Tom Kazakov
# RBE 549 Week 7 Assignment:  Coin detection, recognition, and tallying

import pickle
from pathlib import Path

import cv2
import numpy as np

# US Treasury coin specifications (diameter in mm)
COIN_SPECS = {
    "penny": {"diameter_mm": 19.05, "value": 0.01},
    "nickel": {"diameter_mm": 21.21, "value": 0.05},
    "dime": {"diameter_mm": 17.91, "value": 0.10},
    "quarter": {"diameter_mm": 24.26, "value": 0.25},
    "dollar": {"diameter_mm": 26.50, "value": 1.00},
}

SMALLEST_COIN_MM = min(s["diameter_mm"] for s in COIN_SPECS.values())
LARGEST_COIN_MM = max(s["diameter_mm"] for s in COIN_SPECS.values())

# Hough detection defaults (pre-calibration)
DEFAULT_MIN_RADIUS = 20
DEFAULT_MAX_RADIUS = 200
HOUGH_PARAM1 = 100
HOUGH_PARAM2 = 40
HOUGH_MIN_DIST_FACTOR = 2.5
MEDIAN_BLUR_K = 5

# Adaptive radius tolerance — fraction of expected radius added as margin
RADIUS_TOLERANCE = 0.25

CAMERA_INDEX = 0
WINDOW_NAME = "VisionCoin"
DATABASE_PATH = Path("VisionCoin.pkl")

CIRCLE_COLOR = (0, 255, 0)
CENTER_COLOR = (0, 0, 255)
CIRCLE_THICKNESS = 2
CENTER_RADIUS = 3

PERSISTED_KEYS = [
    "hough_param1",
    "hough_param2",
    "hough_min_radius",
    "hough_max_radius",
    "scale_factor",
]


def create_state():
    """Return initial application state."""
    return {
        "mode": "detect",
        "scale_factor": None,
        "hough_param1": HOUGH_PARAM1,
        "hough_param2": HOUGH_PARAM2,
        "hough_min_radius": DEFAULT_MIN_RADIUS,
        "hough_max_radius": DEFAULT_MAX_RADIUS,
    }


def save_database(state):
    """Persist calibration and learned data to disk."""
    data = {k: state[k] for k in PERSISTED_KEYS if k in state}
    with open(DATABASE_PATH, "wb") as f:
        pickle.dump(data, f)


def load_database():
    """Load persisted data if available."""
    if not DATABASE_PATH.exists():
        return {}
    with open(DATABASE_PATH, "rb") as f:
        return pickle.load(f)


def adaptive_radius_bounds(scale_factor):
    """Compute tight Hough radius bounds from calibration scale factor."""
    min_r = int((SMALLEST_COIN_MM / 2) * scale_factor * (1 - RADIUS_TOLERANCE))
    max_r = int((LARGEST_COIN_MM / 2) * scale_factor * (1 + RADIUS_TOLERANCE))
    return max(1, min_r), max_r


def setup_trackbars(window_name, state):
    """Create trackbars for tuning Hough parameters."""

    def make_callback(key):
        def callback(val):
            state[key] = val
            save_database(state)

        return callback

    cv2.createTrackbar(
        "param1", window_name, state["hough_param1"], 300, make_callback("hough_param1")
    )
    cv2.createTrackbar(
        "param2", window_name, state["hough_param2"], 150, make_callback("hough_param2")
    )
    cv2.createTrackbar(
        "minRadius",
        window_name,
        state["hough_min_radius"],
        300,
        make_callback("hough_min_radius"),
    )
    cv2.createTrackbar(
        "maxRadius",
        window_name,
        state["hough_max_radius"],
        500,
        make_callback("hough_max_radius"),
    )


def remove_trackbars(window_name):
    """Destroy and recreate the window to remove trackbars."""
    cv2.destroyWindow(window_name)
    cv2.namedWindow(window_name)


def detect_coins(frame, state):
    """Detect circular objects via Hough Circle Transform."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, MEDIAN_BLUR_K)

    if state["scale_factor"] is not None:
        min_r, max_r = adaptive_radius_bounds(state["scale_factor"])
    else:
        min_r = state["hough_min_radius"]
        max_r = state["hough_max_radius"]

    p1 = max(1, state["hough_param1"])
    p2 = max(1, state["hough_param2"])
    min_dist = max(1, int(min_r * HOUGH_MIN_DIST_FACTOR))

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_dist,
        param1=p1,
        param2=p2,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is None:
        return []
    return np.round(circles[0]).astype(int).tolist()


def draw_circles(frame, circles):
    """Draw detected circles on frame."""
    for x, y, r in circles:
        cv2.circle(frame, (x, y), r, CIRCLE_COLOR, CIRCLE_THICKNESS)
        cv2.circle(frame, (x, y), CENTER_RADIUS, CENTER_COLOR, -1)


def draw_overlay(frame, state, circles):
    """Draw mode and detection count on frame."""
    mode_text = f"Mode: {state['mode']}"
    count_text = f"Detected: {len(circles)}"
    cal_text = "Calibrated" if state["scale_factor"] else "Uncalibrated"

    cv2.putText(
        frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    cv2.putText(
        frame, count_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    cv2.putText(
        frame, cal_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )

    if state["scale_factor"] is not None:
        min_r, max_r = adaptive_radius_bounds(state["scale_factor"])
        bounds_text = f"Radius bounds: {min_r}-{max_r}px"
        cv2.putText(
            frame,
            bounds_text,
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )


def handle_key(key, state):
    """Process keyboard input, return True if app should quit."""
    if key == 27:
        return True
    if key == ord("t"):
        if state["mode"] == "tune":
            state["mode"] = "detect"
            remove_trackbars(WINDOW_NAME)
        else:
            state["mode"] = "tune"
            setup_trackbars(WINDOW_NAME, state)
    return False


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: cannot open camera")
        return

    state = create_state()
    persisted = load_database()
    state.update(persisted)

    cv2.namedWindow(WINDOW_NAME)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        circles = detect_coins(frame, state)
        draw_circles(frame, circles)
        draw_overlay(frame, state, circles)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if handle_key(key, state):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
