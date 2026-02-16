# Tom Kazakov
# RBE 549 Lab 5: Feature Matching & Object Detection
#
# SURF requires OpenCV built with NONFREE flag:
# CMAKE_ARGS="-DOPENCV_ENABLE_NONFREE=ON" pip install --no-binary=opencv-contrib-python opencv-contrib-python

import cv2
import numpy as np

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

# Homography
MIN_MATCH_COUNT = 10
RANSAC_REPROJ_THRESHOLD = 5.0

# Visualization
USE_GRAYSCALE = True
INLIER_COLOR = (0, 255, 0)
OUTLIER_COLOR = (0, 0, 255)
BOX_COLOR = (255, 0, 0)
BOX_THICKNESS = 2


def detect_sift(img):
    """Detect SIFT keypoints and compute descriptors."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(gray, None)


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
    detect_fn = {"SIFT": detect_sift}[detector_name]
    match_fn = {"BF": match_bruteforce, "FLANN": match_flann}[matcher_name]

    query_kp, query_des = detect_fn(query_img)
    scene_kp, scene_des = detect_fn(scene_img)
    good = match_fn(query_des, scene_des)
    result = find_object(query_img, query_kp, scene_img, scene_kp, good)
    return result, len(good)


USE_FLANN = True

if __name__ == "__main__":
    book = cv2.imread(BOOK_PATH)
    table = cv2.imread(TABLE_PATH)

    matcher_name = "FLANN" if USE_FLANN else "BF"
    result, count = run_combination("SIFT", matcher_name, book, table)

    label = f"SIFT + {matcher_name} | ratio {RATIO_THRESHOLD} | {count} matches"
    cv2.putText(result, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, INLIER_COLOR, 2)

    print(label)
    cv2.imshow(f"SIFT + {matcher_name}", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
