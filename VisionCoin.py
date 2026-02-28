# Tom Kazakov
# RBE 549 Week 7 Assignment:  Coin detection, recognition, and tallying

import pickle
from collections import Counter, deque
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
DEFAULT_MIN_RADIUS = 80
DEFAULT_MAX_RADIUS = 300
HOUGH_PARAM1 = 290
HOUGH_PARAM2 = 60
HOUGH_MIN_DIST_FACTOR = 2.5
MEDIAN_BLUR_K = 5

# Adaptive radius tolerance — fraction of expected radius added as margin
RADIUS_TOLERANCE = 0.10

# Half the smallest diameter gap (penny–dime: 1.14mm) prevents classification overlap
SIZE_TOLERANCE_MM = 1.0

# Color classification — ROI mask shrink and HSV thresholds
ROI_SHRINK = 0.80
SATURATION_THRESHOLD = 50
COPPER_HUE_RANGE = (10, 18)
GOLD_HUE_RANGE = (19, 26)

# Temporal smoothing
WINDOW_SIZE = 15
MIN_OBSERVATIONS = 5
MATCH_DISTANCE = 50
EVICTION_FRAMES = 5

# SIFT/FLANN feature matching
FLANN_INDEX_KDTREE = 1
FLANN_TREES = 5
FLANN_CHECKS = 100
RATIO_THRESHOLD = 0.75
MIN_GOOD_MATCHES = 5
SIFT_DISCRIMINATION_RATIO = 1.5
SIFT_VOTE_WEIGHT = 3

CAMERA_INDEX = 0
WINDOW_NAME = "VisionCoin"
DATABASE_PATH = Path("VisionCoin.pkl")
FEATURES_PATH = Path("VisionCoin_features.pkl")

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
        "mode": "tally",
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
        300,
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


def extract_coin_roi(frame, x, y, r):
    """Extract circular ROI pixels from a coin, shrunk to avoid edge artifacts."""
    h, w = frame.shape[:2]
    inner_r = int(r * ROI_SHRINK)
    x1, y1 = max(0, x - inner_r), max(0, y - inner_r)
    x2, y2 = min(w, x + inner_r), min(h, y + inner_r)
    crop = frame[y1:y2, x1:x2]
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (x - x1, y - y1), inner_r, 255, -1)
    return crop, mask


def classify_by_color(frame, circles):
    """Classify coins by HSV color, returning labels and raw diagnostics."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    results = []
    for x, y, r in circles:
        crop, mask = extract_coin_roi(hsv, x, y, r)
        mean_hue = cv2.mean(crop[:, :, 0], mask=mask)[0]
        mean_sat = cv2.mean(crop[:, :, 1], mask=mask)[0]
        if mean_sat < SATURATION_THRESHOLD:
            label = "silver"
        elif COPPER_HUE_RANGE[0] <= mean_hue <= COPPER_HUE_RANGE[1]:
            label = "copper"
        elif GOLD_HUE_RANGE[0] <= mean_hue <= GOLD_HUE_RANGE[1]:
            label = "gold"
        else:
            label = "silver"
        results.append({"label": label, "hue": mean_hue, "sat": mean_sat})
    return results


def fuse_size_and_color(size_labels, color_labels):
    """Combine size and color — color only resolves the penny/dime ambiguity."""
    fused = []
    for size_l, color_l in zip(size_labels, color_labels):
        if size_l is None:
            fused.append(None)
        elif size_l in ("penny", "dime"):
            if color_l == "copper":
                fused.append("penny")
            elif color_l == "silver":
                fused.append("dime")
            else:
                fused.append(size_l)
        else:
            fused.append(size_l)
    return fused


def save_features(feature_db):
    """Persist learned feature database to disk."""
    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(feature_db, f)


def load_features():
    """Load learned feature database if available."""
    if not FEATURES_PATH.exists():
        return {}
    with open(FEATURES_PATH, "rb") as f:
        return pickle.load(f)


def extract_full_roi(frame, x, y, r):
    """Extract circular ROI at full detected radius for feature extraction."""
    h, w = frame.shape[:2]
    x1, y1 = max(0, x - r), max(0, y - r)
    x2, y2 = min(w, x + r), min(h, y + r)
    crop = frame[y1:y2, x1:x2]
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (x - x1, y - y1), r, 255, -1)
    return crop, mask


SIFT = cv2.SIFT_create()


def extract_features(frame, x, y, r):
    """Extract SIFT keypoints and descriptors from a circular coin ROI."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    crop, mask = extract_full_roi(gray, x, y, r)
    kp, des = SIFT.detectAndCompute(crop, mask)
    return kp, des


def learn_coins(frame, circles, denomination, feature_db):
    """Learn SIFT features from all detected coins."""
    if denomination not in feature_db:
        feature_db[denomination] = []
    learned = 0
    for x, y, r in circles:
        kp, des = extract_features(frame, x, y, r)
        if des is not None and len(des) > 0:
            feature_db[denomination].append(des)
            learned += 1
    if learned > 0:
        save_features(feature_db)
    return learned


def classify_single_coin(frame, x, y, r, feature_db):
    """Run SIFT/FLANN for one coin, returning (bias_label, top_match, keypoints)."""
    kp, des = extract_features(frame, x, y, r)
    if des is None or len(des) < 2:
        return None, None, (kp, x, y, r)

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=FLANN_TREES)
    search_params = dict(checks=FLANN_CHECKS)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    match_counts = {}
    for name, samples in feature_db.items():
        total_good = 0
        for sample in samples:
            stored_des = sample["des"] if isinstance(sample, dict) else sample
            if stored_des is None or len(stored_des) < 2:
                continue
            matches = flann.knnMatch(des, stored_des, k=2)
            good = [m for m, n in matches if m.distance < RATIO_THRESHOLD * n.distance]
            total_good += len(good)
        match_counts[name] = total_good

    ranked = sorted(match_counts.items(), key=lambda item: item[1], reverse=True)
    best_name, best_count = ranked[0]
    second_count = ranked[1][1] if len(ranked) > 1 else 0

    counts_str = " ".join(f"{n}:{c}" for n, c in ranked)
    ratio = best_count / second_count if second_count > 0 else float("inf")
    print(f"SIFT @ ({x},{y}): {counts_str}  ratio={ratio:.2f}")

    top_match = best_name if best_count >= MIN_GOOD_MATCHES else None

    if len(ranked) < 2 or best_count < MIN_GOOD_MATCHES:
        bias_label = None
    elif second_count == 0 or best_count > second_count * SIFT_DISCRIMINATION_RATIO:
        bias_label = best_name
    else:
        bias_label = None
    return bias_label, top_match, (kp, x, y, r)


def draw_keypoints_on_frame(frame, all_keypoints):
    """Draw SIFT keypoint markers on the main frame."""
    for kp, cx, cy, r in all_keypoints:
        if kp is None:
            continue
        ox, oy = cx - r, cy - r
        for pt in kp:
            px = int(pt.pt[0] + ox)
            py = int(pt.pt[1] + oy)
            cv2.circle(frame, (px, py), 2, (255, 0, 255), -1)


class CoinTracker:
    """Temporal smoothing via per-coin sliding windows with SIFT queue."""

    def __init__(self):
        self._tracks = {}
        self._next_id = 0
        self._frame_count = 0
        self._sift_queue = deque()

    def update(self, circles, coin_info):
        """Match detections to existing tracks, create/evict as needed."""
        self._frame_count += 1
        matched_ids = set()
        unmatched = list(range(len(circles)))

        for track_id, track in list(self._tracks.items()):
            if not unmatched:
                break
            tx = int(np.median([o["x"] for o in track["observations"]]))
            ty = int(np.median([o["y"] for o in track["observations"]]))
            best_idx, best_dist = None, float("inf")
            for idx in unmatched:
                x, y, r = circles[idx]
                dist = ((x - tx) ** 2 + (y - ty) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_dist < MATCH_DISTANCE:
                prev_label = self._stable_label(track)
                self._add_observation(
                    track, circles[best_idx], coin_info[best_idx] if coin_info else {}
                )
                track["last_seen"] = self._frame_count
                matched_ids.add(track_id)
                unmatched.remove(best_idx)
                new_label = self._stable_label(track)
                if new_label and new_label != prev_label:
                    self._enqueue_sift(track_id)

        for idx in unmatched:
            info = coin_info[idx] if coin_info else {}
            self._create_track(circles[idx], info)

        for track_id in list(self._tracks):
            if (
                track_id not in matched_ids
                and self._frame_count - self._tracks[track_id]["last_seen"]
                > EVICTION_FRAMES
            ):
                del self._tracks[track_id]

        self._sift_queue = deque(tid for tid in self._sift_queue if tid in self._tracks)

    def _stable_label(self, track):
        """Return majority-vote label if track has enough observations."""
        obs = track["observations"]
        if len(obs) < MIN_OBSERVATIONS:
            return None
        labels = [o["label"] for o in obs if o["label"]]
        if not labels:
            return None
        return Counter(labels).most_common(1)[0][0]

    def _enqueue_sift(self, track_id):
        if track_id not in self._sift_queue:
            self._sift_queue.append(track_id)

    def pop_sift_target(self):
        """Return (track_id, x, y, r) for the next coin needing SIFT, or None."""
        while self._sift_queue:
            track_id = self._sift_queue.popleft()
            if track_id not in self._tracks:
                continue
            track = self._tracks[track_id]
            obs = track["observations"]
            x = int(np.median([o["x"] for o in obs]))
            y = int(np.median([o["y"] for o in obs]))
            r = int(np.median([o["r"] for o in obs]))
            return track_id, x, y, r
        return None

    def set_sift_result(self, track_id, bias_label, top_match, keypoints):
        """Store SIFT classification result on a track."""
        if track_id in self._tracks:
            self._tracks[track_id]["sift_label"] = bias_label
            self._tracks[track_id]["sift_top"] = top_match
            self._tracks[track_id]["sift_keypoints"] = keypoints

    def _create_track(self, circle, info):
        track = {
            "observations": deque(maxlen=WINDOW_SIZE),
            "last_seen": self._frame_count,
            "sift_label": None,
            "sift_top": None,
            "sift_keypoints": None,
        }
        self._add_observation(track, circle, info)
        self._tracks[self._next_id] = track
        self._next_id += 1

    def _add_observation(self, track, circle, info):
        x, y, r = circle
        track["observations"].append(
            {
                "x": x,
                "y": y,
                "r": r,
                "label": info.get("label"),
                "color": info.get("color"),
                "size_mm": info.get("size_mm"),
                "hsv": info.get("hsv"),
            }
        )

    def get_stable_coins(self):
        """Return smoothed circles and info for tracks with enough observations."""
        stable_circles = []
        stable_info = []
        for track in self._tracks.values():
            obs = track["observations"]
            if len(obs) < MIN_OBSERVATIONS:
                continue
            x = int(np.median([o["x"] for o in obs]))
            y = int(np.median([o["y"] for o in obs]))
            r = int(np.median([o["r"] for o in obs]))
            stable_circles.append([x, y, r])

            labels = [o["label"] for o in obs if o["label"]]
            colors = [o["color"] for o in obs if o["color"]]
            sizes = [o["size_mm"] for o in obs if o["size_mm"]]
            hsvs = [o["hsv"] for o in obs if o["hsv"]]

            sift_label = track.get("sift_label")
            if sift_label:
                labels.extend([sift_label] * SIFT_VOTE_WEIGHT)

            info = {}
            if labels:
                info["label"] = Counter(labels).most_common(1)[0][0]
            if colors:
                info["color"] = Counter(colors).most_common(1)[0][0]
            if sizes:
                info["size_mm"] = Counter(sizes).most_common(1)[0][0]
            if hsvs:
                info["hsv"] = hsvs[-1]
            sift_top = track.get("sift_top")
            if sift_top:
                info["sift"] = sift_top
            stable_info.append(info)

        return stable_circles, stable_info

    def get_all_sift_keypoints(self):
        """Return cached SIFT keypoints from all tracks that have them."""
        keypoints = []
        for track in self._tracks.values():
            kp_data = track.get("sift_keypoints")
            if kp_data:
                keypoints.append(kp_data)
        return keypoints

    def reset(self):
        """Clear all tracks and SIFT queue."""
        self._tracks.clear()
        self._next_id = 0
        self._sift_queue.clear()


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
    cv2.line(
        frame,
        (cx - CROSSHAIR_SIZE, cy),
        (cx - CROSSHAIR_GAP, cy),
        CROSSHAIR_COLOR,
        CROSSHAIR_THICKNESS,
    )
    cv2.line(
        frame,
        (cx + CROSSHAIR_GAP, cy),
        (cx + CROSSHAIR_SIZE, cy),
        CROSSHAIR_COLOR,
        CROSSHAIR_THICKNESS,
    )
    cv2.line(
        frame,
        (cx, cy - CROSSHAIR_SIZE),
        (cx, cy - CROSSHAIR_GAP),
        CROSSHAIR_COLOR,
        CROSSHAIR_THICKNESS,
    )
    cv2.line(
        frame,
        (cx, cy + CROSSHAIR_GAP),
        (cx, cy + CROSSHAIR_SIZE),
        CROSSHAIR_COLOR,
        CROSSHAIR_THICKNESS,
    )


def _draw_text_with_bg(frame, text, x, y, scale, thickness, alpha=0.5):
    """Draw horizontally centered text with a translucent dark backing."""
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    pad = 2
    x1 = x - tw // 2 - pad
    y1 = y - th - pad
    x2 = x + tw // 2 + pad
    y2 = y + baseline + pad
    overlay = frame[max(0, y1) : y2, max(0, x1) : x2]
    if overlay.size > 0:
        dark = np.zeros_like(overlay)
        cv2.addWeighted(overlay, 1 - alpha, dark, alpha, 0, overlay)
    cv2.putText(
        frame,
        text,
        (x - tw // 2, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
    )


def draw_circles(frame, circles, coin_info=None):
    """Draw detected circles on frame with diagnostic and denomination labels."""
    for i, (x, y, r) in enumerate(circles):
        cv2.circle(frame, (x, y), r, CIRCLE_COLOR, CIRCLE_THICKNESS)
        cv2.circle(frame, (x, y), CENTER_RADIUS, CENTER_COLOR, -1)
        if not coin_info:
            continue
        info = coin_info[i]
        denom = info.get("label")
        size_text = info.get("size_mm", "")
        color_text = info.get("color", "")
        hsv_text = info.get("hsv", "")
        sift_text = info.get("sift", "")
        line_y = y - r // 3
        if size_text:
            _draw_text_with_bg(frame, size_text, x, line_y, 0.4, 1)
            line_y += 16
        if color_text:
            _draw_text_with_bg(frame, color_text, x, line_y, 0.4, 1)
            line_y += 16
        if hsv_text:
            _draw_text_with_bg(frame, hsv_text, x, line_y, 0.35, 1)
            line_y += 14
        if sift_text:
            _draw_text_with_bg(frame, f"sift:{sift_text}", x, line_y, 0.35, 1)
        if denom:
            _draw_text_with_bg(frame, denom, x, y + r // 4, 0.5, 2)
            value_text = f"${COIN_SPECS[denom]['value']:.2f}"
            _draw_text_with_bg(frame, value_text, x, y + r // 4 + 20, 0.5, 2)


def draw_calibration(frame, state, circles):
    """Draw calibration-specific UI elements."""
    draw_crosshairs(frame)

    denom = state["selected_denomination"]
    if denom:
        spec = COIN_SPECS[denom]
        prompt = f"Place {denom} (${spec['value']:.2f}) at crosshairs, press Space"
    else:
        prompt = "Select coin: 1=penny 2=nickel 3=dime 4=quarter 5=dollar"

    cv2.putText(
        frame,
        prompt,
        (10, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    if denom and circles:
        h, w = frame.shape[:2]
        idx = nearest_to_center(circles, w, h)
        x, y, r = circles[idx]
        cv2.circle(frame, (x, y), r, HIGHLIGHT_COLOR, 3)


def draw_learning(frame, state, circles, feature_db):
    """Draw learning mode UI elements."""
    denom = state["selected_denomination"]
    sample_counts = {name: len(samples) for name, samples in feature_db.items()}
    counts_text = (
        "  ".join(f"{n}:{c}" for n, c in sample_counts.items())
        if sample_counts
        else "empty"
    )

    if denom:
        prompt = (
            f"Place {denom}s in view, press Space to learn ({len(circles)} detected)"
        )
    else:
        prompt = "Select coin: 1=penny 2=nickel 3=dime 4=quarter 5=dollar"

    cv2.putText(
        frame,
        prompt,
        (10, frame.shape[0] - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Learned: {counts_text}",
        (10, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )

    if denom and circles:
        for x, y, r in circles:
            cv2.circle(frame, (x, y), r, HIGHLIGHT_COLOR, 3)


def compute_tally(stable_info):
    """Sum dollar values of all recognized coins."""
    total = 0.0
    for info in stable_info:
        label = info.get("label")
        if label and label in COIN_SPECS:
            total += COIN_SPECS[label]["value"]
    return total


def _draw_overlay_bg(frame, line_count):
    """Draw a translucent dark strip behind the overlay text area."""
    h = 10 + line_count * 30
    overlay = frame[0:h, 0 : frame.shape[1] // 3]
    if overlay.size > 0:
        dark = np.zeros_like(overlay)
        cv2.addWeighted(overlay, 0.5, dark, 0.5, 0, overlay)


def draw_overlay(frame, state, circles, stable_info=None):
    """Draw mode, detection count, and tally on frame."""
    lines = ["mode", "count", "cal"]
    if state["scale_factor"] is not None:
        lines.append("bounds")
    if stable_info:
        lines.append("tally")
    _draw_overlay_bg(frame, len(lines))

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

    y_offset = 120
    if state["scale_factor"] is not None:
        min_r, max_r = adaptive_radius_bounds(state["scale_factor"])
        bounds_text = f"Radius bounds: {min_r}-{max_r}px"
        cv2.putText(
            frame,
            bounds_text,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )
        y_offset += 30

    if stable_info:
        total = compute_tally(stable_info)
        tally_text = f"Total: ${total:.2f}"
        cv2.putText(
            frame,
            tally_text,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )


def _leave_tune_if_active(state):
    """Remove trackbars when leaving tune mode."""
    if state["mode"] == "tune":
        remove_trackbars(WINDOW_NAME)


def handle_key(key, state):
    """Process keyboard input, return True if app should quit."""
    if key == 27:
        return True

    if key == ord("t"):
        if state["mode"] == "tune":
            state["mode"] = "tally"
            remove_trackbars(WINDOW_NAME)
        else:
            _leave_tune_if_active(state)
            state["mode"] = "tune"
            setup_trackbars(WINDOW_NAME, state)
        return False

    if key == ord("c"):
        _leave_tune_if_active(state)
        state["mode"] = "calibrate"
        state["selected_denomination"] = None
        state["scale_factor"] = None
        return False

    if key == ord("l"):
        if state["mode"] == "learn":
            state["mode"] = "tally"
            state["selected_denomination"] = None
        else:
            _leave_tune_if_active(state)
            state["mode"] = "learn"
            state["selected_denomination"] = None
        return False

    if key == ord("f"):
        if state["mode"] == "sift":
            state["mode"] = "tally"
        else:
            _leave_tune_if_active(state)
            state["mode"] = "sift"
        return False

    if key in DENOMINATION_KEYS and state["mode"] in ("calibrate", "learn"):
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
    state["mode"] = "tally"
    save_database(state)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: cannot open camera")
        return

    state = create_state()
    persisted = load_database()
    state.update(persisted)
    feature_db = load_features()
    tracker = CoinTracker()

    cv2.namedWindow(WINDOW_NAME)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        circles = detect_coins(frame, state)

        coin_info = None
        if state["mode"] in ("tally", "sift") and circles:
            color_results = classify_by_color(frame, circles)
            color_labels = [r["label"] for r in color_results]
            if state["scale_factor"] is not None:
                size_labels = classify_by_size(circles, state["scale_factor"])
                fused = fuse_size_and_color(size_labels, color_labels)
            else:
                fused = [None] * len(circles)
            coin_info = []
            for i, (x, y, r) in enumerate(circles):
                info = {
                    "label": fused[i],
                    "color": color_labels[i],
                    "hsv": f"H:{color_results[i]['hue']:.0f} S:{color_results[i]['sat']:.0f}",
                }
                if state["scale_factor"] is not None:
                    info["size_mm"] = f"{2 * r * state['scale_factor']:.1f}mm"
                coin_info.append(info)

        display_info = None
        if state["mode"] in ("calibrate", "tune", "learn"):
            tracker.reset()
            draw_circles(frame, circles)
            display_circles = circles
        else:
            tracker.update(circles, coin_info or [])
            if feature_db:
                target = tracker.pop_sift_target()
                if target:
                    tid, tx, ty, tr = target
                    bias_label, top_match, kp_data = classify_single_coin(
                        frame, tx, ty, tr, feature_db
                    )
                    tracker.set_sift_result(tid, bias_label, top_match, kp_data)
            stable_circles, stable_info = tracker.get_stable_coins()
            draw_circles(frame, stable_circles, stable_info or None)
            display_circles = stable_circles
            display_info = stable_info

        if state["mode"] == "calibrate":
            draw_calibration(frame, state, circles)
        elif state["mode"] == "learn":
            draw_learning(frame, state, circles, feature_db)
        elif state["mode"] == "sift":
            all_keypoints = tracker.get_all_sift_keypoints()
            if all_keypoints:
                draw_keypoints_on_frame(frame, all_keypoints)

        draw_overlay(frame, state, display_circles, display_info)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if handle_key(key, state):
            break

        if state.get("capture_requested"):
            state["capture_requested"] = False
            if state["mode"] == "calibrate":
                process_calibration(state, circles, frame.shape)
            elif state["mode"] == "learn":
                denom = state["selected_denomination"]
                if denom and circles:
                    count = learn_coins(frame, circles, denom, feature_db)
                    print(f"Learned {count} {denom} samples ({len(circles)} detected)")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
