# Tom Kazakov
# RBE 549 Lab 6: Panorama Stitching

import cv2
import numpy as np

BOSTON1_PATH = "boston1.jpeg"
BOSTON2_PATH = "boston2.jpeg"
PANORAMA_OUTPUT_PATH = "panorama.jpg"

# Lowe's ratio test
RATIO_THRESHOLD = 0.75

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
INLIER_COLOR = (0, 255, 0)
OUTLIER_COLOR = (0, 0, 255)


def detect_keypoints(img):
    """Detect SIFT keypoints and compute descriptors."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(gray, None)


def match_descriptors(des1, des2):
    """Match descriptors with FLANN + Lowe's ratio test."""
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=FLANN_TREES)
    search_params = dict(checks=FLANN_CHECKS)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    raw = flann.knnMatch(des1, des2, k=KNN_NEIGHBORS)
    return [m for m, n in raw if m.distance < RATIO_THRESHOLD * n.distance]


def compute_homography(kp1, kp2, matches):
    """Estimate homography from img2 to img1 via RANSAC."""
    src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    return cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)


def draw_matches(img1, kp1, img2, kp2, matches, mask):
    """Draw side-by-side match visualization with green inliers and red outliers."""
    h1, w1 = img1.shape[:2]
    canvas = np.hstack([img1, img2])
    mask_flat = mask.ravel()
    for i, m in enumerate(matches):
        pt1 = tuple(np.int32(kp1[m.queryIdx].pt))
        pt2_raw = np.int32(kp2[m.trainIdx].pt)
        pt2 = (pt2_raw[0] + w1, pt2_raw[1])
        color = INLIER_COLOR if mask_flat[i] else OUTLIER_COLOR
        cv2.line(canvas, pt1, pt2, color, 1)
    inlier_count = int(mask.sum())
    outlier_count = len(matches) - inlier_count
    cv2.putText(
        canvas,
        f"Inliers: {inlier_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        INLIER_COLOR,
        2,
    )
    cv2.putText(
        canvas,
        f"Outliers: {outlier_count}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        OUTLIER_COLOR,
        2,
    )
    return canvas


if __name__ == "__main__":
    img1 = cv2.imread(BOSTON1_PATH)
    img2 = cv2.imread(BOSTON2_PATH)

    kp1, des1 = detect_keypoints(img1)
    kp2, des2 = detect_keypoints(img2)
    print(f"Keypoints: img1={len(kp1)}, img2={len(kp2)}")

    matches = match_descriptors(des1, des2)
    print(f"Good matches after ratio test: {len(matches)}")

    H, mask = compute_homography(kp1, kp2, matches)
    print(f"Inliers: {int(mask.sum())} / {len(matches)}")

    vis = draw_matches(img1, kp1, img2, kp2, matches, mask)
    cv2.imshow("Matches", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
