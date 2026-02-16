# Tom Kazakov
# RBE 549 Lab 5: Feature Matching & Object Detection
#
# SURF requires OpenCV built with NONFREE flag:
# CMAKE_ARGS="-DOPENCV_ENABLE_NONFREE=ON" pip install --no-binary=opencv-contrib-python opencv-contrib-python

import time

import cv2
import numpy as np
from pathlib import Path

BOOK_PATH = "book.jpg"
TABLE_PATH = "table.jpg"

# Lowe's ratio test
RATIO_THRESHOLD = 0.5

# Matching
KNN_NEIGHBORS = 2

# FLANN index
FLANN_INDEX_KDTREE = 1
FLANN_TREES = 5
FLANN_CHECKS = 100

# SURF
SURF_HESSIAN_THRESHOLD = 400

# Homography
MIN_MATCH_COUNT = 10
RANSAC_REPROJ_THRESHOLD = 5.0

# Visualization
USE_GRAYSCALE = True
INLIER_COLOR = (0, 255, 0)
OUTLIER_COLOR = (0, 0, 255)
BOX_COLOR = (255, 0, 0)
BOX_THICKNESS = 2

# Camera real-time detection
CAMERA_DETECTOR = "SIFT"
CAMERA_MATCHER = "FLANN"
BENCHMARK = True
BENCHMARK_DURATION = 5.0
REFERENCE_PATH = "captures/reference.jpg"
THUMBNAIL_WIDTH = 300
THUMBNAIL_BORDER = 2
THUMBNAIL_ALPHA = 0.7
THUMBNAIL_MARGIN = 10


def detect_sift(img):
    """Detect SIFT keypoints and compute descriptors."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(gray, None)


def detect_surf(img):
    """Detect SURF keypoints and compute descriptors."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    surf = cv2.xfeatures2d.SURF_create(SURF_HESSIAN_THRESHOLD)
    return surf.detectAndCompute(gray, None)


def match_bruteforce(des1, des2):
    """Match descriptors with BFMatcher + Lowe's ratio test."""
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(des1, des2, k=KNN_NEIGHBORS)
    return [m for m, n in raw if m.distance < RATIO_THRESHOLD * n.distance]


def match_flann(des1, des2):
    """Match descriptors with FLANN + Lowe's ratio test."""
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=FLANN_TREES)
    search_params = dict(checks=FLANN_CHECKS)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    raw = flann.knnMatch(des1, des2, k=KNN_NEIGHBORS)
    return [m for m, n in raw if m.distance < RATIO_THRESHOLD * n.distance]


def _to_gray_bgr(img):
    """Convert to grayscale and back to BGR for consistent visualization."""
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def find_object(query_img, query_kp, scene_img, scene_kp, matches):
    """Draw homography bounding box, inlier/outlier match lines, and legend."""
    h, w = query_img.shape[:2]
    q = _to_gray_bgr(query_img) if USE_GRAYSCALE else query_img
    s = _to_gray_bgr(scene_img) if USE_GRAYSCALE else scene_img

    if len(matches) < MIN_MATCH_COUNT:
        return cv2.drawMatches(
            q,
            query_kp,
            s,
            scene_kp,
            matches,
            None,
            matchColor=OUTLIER_COLOR,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

    src_pts = np.float32([query_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([scene_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)

    if M is None:
        return cv2.drawMatches(
            q,
            query_kp,
            s,
            scene_kp,
            matches,
            None,
            matchColor=OUTLIER_COLOR,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

    # Build side-by-side canvas, then draw match lines manually by inlier status.
    result = np.hstack([q, s])
    mask_flat = mask.ravel()
    for i, m in enumerate(matches):
        pt1 = tuple(np.int32(query_kp[m.queryIdx].pt))
        pt2_raw = np.int32(scene_kp[m.trainIdx].pt)
        pt2 = (pt2_raw[0] + w, pt2_raw[1])
        color = INLIER_COLOR if mask_flat[i] else OUTLIER_COLOR
        cv2.line(result, pt1, pt2, color, 1)

    # Blue frame on query image (left side).
    cv2.rectangle(result, (0, 0), (w - 1, h - 1), BOX_COLOR, BOX_THICKNESS)

    # Blue frame on detected object in scene (right side).
    query_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    scene_corners = cv2.perspectiveTransform(query_corners, M)
    offset = np.float32([w, 0]).reshape(1, 1, 2)
    cv2.polylines(
        result, [np.int32(scene_corners + offset)], True, BOX_COLOR, BOX_THICKNESS
    )

    # Legend.
    inlier_count = int(mask.sum())
    outlier_count = len(matches) - inlier_count
    lx, ly = 10, result.shape[0] - 60
    cv2.putText(
        result,
        f"Inliers: {inlier_count}",
        (lx, ly),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        INLIER_COLOR,
        2,
    )
    cv2.putText(
        result,
        f"Outliers: {outlier_count}",
        (lx, ly + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        OUTLIER_COLOR,
        2,
    )
    cv2.putText(
        result,
        "Detection boundary",
        (lx, ly + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        BOX_COLOR,
        2,
    )

    return result


def run_combination(detector_name, matcher_name, query_img, scene_img):
    """Run one detector+matcher combination and return the annotated image + match count."""
    detect_fn = {"SIFT": detect_sift, "SURF": detect_surf}[detector_name]
    match_fn = {"BF": match_bruteforce, "FLANN": match_flann}[matcher_name]

    query_kp, query_des = detect_fn(query_img)
    scene_kp, scene_des = detect_fn(scene_img)
    good = match_fn(query_des, scene_des)
    result = find_object(query_img, query_kp, scene_img, scene_kp, good)
    return result, len(good)


# Camera integration
def _camera_detect_fn():
    """Return the detector function selected by CAMERA_DETECTOR."""
    return {"SIFT": detect_sift, "SURF": detect_surf}[CAMERA_DETECTOR]


def _camera_match_fn():
    """Return the matcher function selected by CAMERA_MATCHER."""
    return {"BF": match_bruteforce, "FLANN": match_flann}[CAMERA_MATCHER]


def init_state():
    """Return Lab 5 state keys with defaults."""
    return {
        "obj_detect_active": False,
        "obj_ref_img": None,
        "obj_ref_kp": None,
        "obj_ref_des": None,
        "obj_bench_start": None,
    }


def setup_trackbars(window_name, state):
    """No trackbars needed for Lab 5."""
    pass


def _capture_reference(frame, state):
    """Save frame as reference and cache keypoints + descriptors."""
    Path("captures").mkdir(exist_ok=True)
    cv2.imwrite(REFERENCE_PATH, frame)
    state["obj_ref_img"] = frame.copy()
    kp, des = _camera_detect_fn()(frame)
    state["obj_ref_kp"] = kp
    state["obj_ref_des"] = des
    state["obj_detect_active"] = False
    print(
        f"[{CAMERA_DETECTOR}] Reference captured → {REFERENCE_PATH} ({len(kp)} keypoints)"
    )


def _load_reference(state):
    """Load persisted reference from disk if not already in memory."""
    if state["obj_ref_img"] is not None:
        return True
    ref = cv2.imread(REFERENCE_PATH)
    if ref is None:
        return False
    state["obj_ref_img"] = ref
    kp, des = _camera_detect_fn()(ref)
    state["obj_ref_kp"] = kp
    state["obj_ref_des"] = des
    print(f"[{CAMERA_DETECTOR}] Reference loaded from disk ({len(kp)} keypoints)")
    return True


def handle_key(key, state, frame):
    """Handle Lab 5 key presses. Returns True if key was handled."""
    if key == ord("o"):
        _capture_reference(frame, state)
        return True
    if key == ord("p"):
        if not _load_reference(state):
            print("No reference — press 'o' to capture")
            return True
        state["obj_detect_active"] = not state["obj_detect_active"]
        status = "ON" if state["obj_detect_active"] else "OFF"
        print(f"Object detection: {status}")
        return True
    return False


def _draw_thumbnail(frame, ref_img):
    """Draw a small reference thumbnail at the mid-left border."""
    rh, rw = ref_img.shape[:2]
    thumb_w = THUMBNAIL_WIDTH
    thumb_h = int(rh * thumb_w / rw)
    thumb = cv2.resize(ref_img, (thumb_w, thumb_h))

    fh, fw = frame.shape[:2]
    x1 = THUMBNAIL_MARGIN
    y1 = (fh - thumb_h - THUMBNAIL_BORDER * 2) // 2
    x2 = x1 + thumb_w + THUMBNAIL_BORDER * 2
    y2 = y1 + thumb_h + THUMBNAIL_BORDER * 2

    if x1 < 0 or y1 < 0 or x2 > fw or y2 > fh:
        return

    frame[y1:y2, x1:x2] = 255
    roi = frame[
        y1 + THUMBNAIL_BORDER : y2 - THUMBNAIL_BORDER,
        x1 + THUMBNAIL_BORDER : x2 - THUMBNAIL_BORDER,
    ]
    cv2.addWeighted(thumb, THUMBNAIL_ALPHA, roi, 1 - THUMBNAIL_ALPHA, 0, roi)


def _detect_on_frame(frame, ref_kp, ref_des, ref_shape):
    """Detect features in frame, match against reference, draw bounding box."""
    frame_kp, frame_des = _camera_detect_fn()(frame)
    if frame_des is None or len(frame_des) < 2:
        return frame

    good = _camera_match_fn()(ref_des, frame_des)
    if len(good) < MIN_MATCH_COUNT:
        return frame

    h, w = ref_shape[:2]
    src_pts = np.float32([ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)

    if M is None:
        return frame

    query_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    scene_corners = cv2.perspectiveTransform(query_corners, M)
    cv2.polylines(frame, [np.int32(scene_corners)], True, BOX_COLOR, BOX_THICKNESS)

    inlier_count = int(mask.sum()) if mask is not None else 0
    label = f"Matches: {len(good)} | Inliers: {inlier_count}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    lx = (frame.shape[1] - tw) // 2
    cv2.putText(
        frame,
        label,
        (lx, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        BOX_COLOR,
        2,
    )
    return frame


def apply_effects(img, state):
    """Apply Lab 5 real-time object detection overlay."""
    if BENCHMARK and state["obj_ref_img"] is None:
        if _load_reference(state):
            state["obj_detect_active"] = True

    if not state["obj_detect_active"] or state["obj_ref_des"] is None:
        return img

    if BENCHMARK:
        if state["obj_bench_start"] is None:
            state["obj_bench_start"] = time.time()
        if time.time() - state["obj_bench_start"] >= BENCHMARK_DURATION:
            raise SystemExit

    result = _detect_on_frame(
        img,
        state["obj_ref_kp"],
        state["obj_ref_des"],
        state["obj_ref_img"].shape,
    )
    _draw_thumbnail(result, state["obj_ref_img"])
    return result


# Standalone execution
USE_SURF = True
USE_FLANN = True

if __name__ == "__main__":
    book = cv2.imread(BOOK_PATH)
    table = cv2.imread(TABLE_PATH)

    detector_name = "SURF" if USE_SURF else "SIFT"
    matcher_name = "FLANN" if USE_FLANN else "BF"
    result, count = run_combination(detector_name, matcher_name, book, table)

    label = (
        f"{detector_name} + {matcher_name} | ratio {RATIO_THRESHOLD} | {count} matches"
    )
    cv2.putText(result, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, INLIER_COLOR, 2)

    print(label)
    cv2.imshow(f"{detector_name} + {matcher_name}", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
