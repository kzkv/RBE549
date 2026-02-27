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

DENOMINATION_NAMES = list(COIN_SPECS.keys())

DENOMINATION_KEYS = {
    ord("1"): "penny",
    ord("2"): "nickel",
    ord("3"): "dime",
    ord("4"): "quarter",
    ord("5"): "dollar",
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

# Size classification tolerance in mm
SIZE_TOLERANCE_MM = 2.0

CAMERA_INDEX = 0
WINDOW_NAME = "VisionCoin"
DATABASE_PATH = Path("VisionCoin.pkl")

CIRCLE_COLOR = (0, 255, 0)
CENTER_COLOR = (0, 0, 255)
HIGHLIGHT_COLOR = (0, 255, 255)
CROSSHAIR_COLOR = (0, 255, 255)
CIRCLE_THICKNESS = 2
CENTER_RADIUS = 3
CROSSHAIR_SIZE = 40
CROSSHAIR_GAP = 8
CROSSHAIR_THICKNESS = 2

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
        "selected_denomination": None,
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
    min_r = int((SMALLEST_COIN_MM / 2) / scale_factor * (1 - RADIUS_TOLERANCE))
    max_r = int((LARGEST_COIN_MM / 2) / scale_factor * (1 + RADIUS_TOLERANCE))
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


def calibrate(pixel_radius, denomination):
    """Compute mm-per-pixel scale factor from a known coin."""
    diameter_mm = COIN_SPECS[denomination]["diameter_mm"]
    return diameter_mm / (2 * pixel_radius)


def expected_radii(scale_factor):
    """Compute expected pixel radius for each denomination."""
    return {
        name: (spec["diameter_mm"] / 2) / scale_factor
        for name, spec in COIN_SPECS.items()
    }


def classify_by_size(circles, scale_factor):
    """Classify each circle by closest denomination based on radius."""
    labels = []
    for x, y, r in circles:
        diameter_mm = 2 * r * scale_factor
        best_name = None
        best_dist = float("inf")
        for name, spec in COIN_SPECS.items():
            dist = abs(diameter_mm - spec["diameter_mm"])
            if dist < best_dist:
                best_dist = dist
                best_name = name
        if best_dist > SIZE_TOLERANCE_MM:
            best_name = None
        labels.append(best_name)
    return labels


def nearest_to_center(circles, frame_w, frame_h):
    """Return index of the circle closest to frame center."""
    cx, cy = frame_w // 2, frame_h // 2
    best_idx = 0
    best_dist = float("inf")
    for i, (x, y, r) in enumerate(circles):
        dist = (x - cx) ** 2 + (y - cy) ** 2
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def draw_crosshairs(frame):
    """Draw crosshairs at frame center with a gap for precise alignment."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - CROSSHAIR_SIZE, cy), (cx - CROSSHAIR_GAP, cy),
             CROSSHAIR_COLOR, CROSSHAIR_THICKNESS)
    cv2.line(frame, (cx + CROSSHAIR_GAP, cy), (cx + CROSSHAIR_SIZE, cy),
             CROSSHAIR_COLOR, CROSSHAIR_THICKNESS)
    cv2.line(frame, (cx, cy - CROSSHAIR_SIZE), (cx, cy - CROSSHAIR_GAP),
             CROSSHAIR_COLOR, CROSSHAIR_THICKNESS)
    cv2.line(frame, (cx, cy + CROSSHAIR_GAP), (cx, cy + CROSSHAIR_SIZE),
             CROSSHAIR_COLOR, CROSSHAIR_THICKNESS)


def draw_circles(frame, circles, labels=None):
    """Draw detected circles on frame with optional denomination labels."""
    for i, (x, y, r) in enumerate(circles):
        cv2.circle(frame, (x, y), r, CIRCLE_COLOR, CIRCLE_THICKNESS)
        cv2.circle(frame, (x, y), CENTER_RADIUS, CENTER_COLOR, -1)
        if labels and labels[i]:
            label = f"{labels[i]} ${COIN_SPECS[labels[i]]['value']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.putText(frame, label, (x - tw // 2, y + r // 2 + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


def draw_calibration(frame, state, circles):
    """Draw calibration-specific UI elements."""
    draw_crosshairs(frame)

    denom = state["selected_denomination"]
    if denom:
        spec = COIN_SPECS[denom]
        prompt = f"Place {denom} (${spec['value']:.2f}) at crosshairs, press Space"
    else:
        prompt = "Select coin: 1=penny 2=nickel 3=dime 4=quarter 5=dollar"

    cv2.putText(frame, prompt, (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if denom and circles:
        h, w = frame.shape[:2]
        idx = nearest_to_center(circles, w, h)
        x, y, r = circles[idx]
        cv2.circle(frame, (x, y), r, HIGHLIGHT_COLOR, 3)


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

    if key == ord("c"):
        state["mode"] = "calibrate"
        state["selected_denomination"] = None
        state["scale_factor"] = None
        return False

    if key in DENOMINATION_KEYS and state["mode"] == "calibrate":
        state["selected_denomination"] = DENOMINATION_KEYS[key]
        return False

    if key == ord(" "):
        state["capture_requested"] = True
        return False

    return False


def process_calibration(state, circles, frame_shape):
    """Execute calibration capture when requested."""
    denom = state["selected_denomination"]
    if not (denom and circles):
        return
    h, w = frame_shape[:2]
    idx = nearest_to_center(circles, w, h)
    pixel_radius = circles[idx][2]
    state["scale_factor"] = calibrate(pixel_radius, denom)
    state["selected_denomination"] = None
    state["mode"] = "detect"
    save_database(state)


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

        labels = None
        if state["scale_factor"] is not None and state["mode"] != "calibrate":
            labels = classify_by_size(circles, state["scale_factor"])

        draw_circles(frame, circles, labels)

        if state["mode"] == "calibrate":
            draw_calibration(frame, state, circles)

        draw_overlay(frame, state, circles)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if handle_key(key, state):
            break

        if state.get("capture_requested"):
            state["capture_requested"] = False
            if state["mode"] == "calibrate":
                process_calibration(state, circles, frame.shape)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
