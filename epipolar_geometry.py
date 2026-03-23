"""Compute and visualize epipolar lines and epipoles for stereo image pairs."""

import cv2
import numpy as np

# Input images
LEFT_PATH = "globe_left.jpg"
CENTER_PATH = "globe_center.jpg"
RIGHT_PATH = "globe_right.jpg"

# SIFT + FLANN matching
RATIO_THRESHOLD = 0.8
FLANN_INDEX_KDTREE = 1
HOMOGRAPHY_THRESHOLD = 5.0

# Epipole x-coordinates for F = [e2]x @ H (planar scene requires explicit placement)
LC_EPIPOLE_X = -10000
CR_EPIPOLE_X = -350

# Visualization
LINE_THICKNESS = 1
POINT_RADIUS = 5


def find_matches(img1, img2):
    """Detect SIFT features and return homography-filtered matches via FLANN."""
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    raw_matches = flann.knnMatch(des1, des2, k=2)

    pts1 = []
    pts2 = []
    for m, n in raw_matches:
        if m.distance < RATIO_THRESHOLD * n.distance:
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, HOMOGRAPHY_THRESHOLD)
    inliers = mask.ravel() == 1
    return pts1[inliers], pts2[inliers], H


def skew_symmetric(e):
    """Build the 3x3 skew-symmetric (cross-product) matrix [e]x."""
    return np.array([[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]])


def compute_F(H, img_shape, epipole_x):
    """Construct F = [e2]x @ H for a planar scene with a specified epipole."""
    h, _ = img_shape[:2]
    e2 = np.array([epipole_x, h / 2, 1.0])
    F = skew_symmetric(e2) @ H
    return F / np.linalg.norm(F)


def draw_epilines(img1, img2, lines, pts1, pts2):
    """Draw epilines on img1 for points in img2, mark points on both images."""
    _, c = img1.shape[:2]
    img1 = img1.copy()
    img2 = img2.copy()
    for line, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [c, -(line[2] + line[0] * c) / line[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, LINE_THICKNESS)
        img1 = cv2.circle(img1, (int(pt1[0]), int(pt1[1])), POINT_RADIUS, color, -1)
        img2 = cv2.circle(img2, (int(pt2[0]), int(pt2[1])), POINT_RADIUS, color, -1)
    return img1, img2


def render_pair(img1, img2, pts1, pts2, F):
    """Draw epilines and corresponding points on both images of a pair."""
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img1_out, img2_out = draw_epilines(img1, img2, lines1, pts1, pts2)

    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img2_out, _ = draw_epilines(img2_out, img1_out, lines2, pts2, pts1)

    return img1_out, img2_out


def process_pair(path1, path2, output_path, epipole_x):
    """Match, compute F from homography, render epilines, and save."""
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    pts1, pts2, H = find_matches(gray1, gray2)
    F = compute_F(H, gray1.shape, epipole_x)

    img1_out, img2_out = render_pair(img1, img2, pts1, pts2, F)
    result = np.hstack([img1_out, img2_out])
    cv2.imwrite(output_path, result)
    print(f"Saved {output_path}")
    return result


if __name__ == "__main__":
    lc = process_pair(LEFT_PATH, CENTER_PATH, "epipolar_lc.jpg", LC_EPIPOLE_X)
    cr = process_pair(CENTER_PATH, RIGHT_PATH, "epipolar_cr.jpg", CR_EPIPOLE_X)

    cv2.imshow("Left + Center", lc)
    cv2.imshow("Center + Right", cr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
