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


if __name__ == "__main__":
    img1 = cv2.imread(BOSTON1_PATH)
    img2 = cv2.imread(BOSTON2_PATH)

    kp1, des1 = detect_keypoints(img1)
    kp2, des2 = detect_keypoints(img2)
    print(f"Keypoints: img1={len(kp1)}, img2={len(kp2)}")

    matches = match_descriptors(des1, des2)
    print(f"Good matches after ratio test: {len(matches)}")
