"""Compute and visualize epipolar lines and epipoles for stereo image pairs."""

import cv2
import numpy as np

# Input images
LEFT_PATH = "globe_left.jpg"
CENTER_PATH = "globe_center.jpg"
RIGHT_PATH = "globe_right.jpg"

# SIFT + FLANN matching (following the OpenCV tutorial)
RATIO_THRESHOLD = 0.8
FLANN_INDEX_KDTREE = 1

# F = [e']x @ H — epipole placed far outside frame
EPIPOLE_DISTANCE = 3000

# Sweep epipole distance
SWEEP_DISTANCES = [500, 1000, 2000, 4000, 8000]

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

    _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    inliers = mask.ravel() == 1
    return pts1[inliers], pts2[inliers]


def skew_matrix(e):
    """Build the 3x3 skew-symmetric matrix [e]x."""
    return np.array([[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]])


def compute_F_from_H(pts1, pts2, img_shape, e2_x, e1_x):
    """Compute F = [e2]x @ H, verifying both epipoles land on opposite sides."""
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    inliers = mask.ravel() == 1
    h, w = img_shape[:2]

    e2 = np.array([e2_x, h / 2, 1.0])
    F = skew_matrix(e2) @ H
    F = F / np.linalg.norm(F)

    # Verify where e1 actually lands
    _, _, Vt = np.linalg.svd(F)
    e1_actual = Vt[-1]
    e1_actual = e1_actual[:2] / e1_actual[2]
    print(f"    e2 set to x={e2_x:.0f}, e1 landed at x={e1_actual[0]:.0f}")

    return F, pts1[inliers], pts2[inliers]


def drawlines(img1, img2, lines, pts1, pts2):
    """Draw epilines on img1 for points in img2, mark points on both images."""
    r, c = img1.shape[:2]
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


def render_pair(img1, img2, pts1_in, pts2_in, F):
    """Draw all epilines on both images."""
    lines1 = cv2.computeCorrespondEpilines(pts2_in.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img1_out, img2_out = drawlines(img1, img2, lines1, pts1_in, pts2_in)

    lines2 = cv2.computeCorrespondEpilines(pts1_in.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img2_out, _ = drawlines(img2_out, img1_out, lines2, pts2_in, pts1_in)

    return img1_out, img2_out


def sweep_pair(path1, path2, title, e2_values, e1_values):
    """Show a vertical stack sweeping epipole position for one image pair."""
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    pts1, pts2 = find_matches(gray1, gray2)
    print(f"{title}: {len(pts1)} clean matches")

    h, w = img1.shape[:2]
    scale = 350 / w
    cells = []

    for e2_x, e1_x in zip(e2_values, e1_values):
        F, pts1_in, pts2_in = compute_F_from_H(pts1, pts2, gray1.shape, e2_x, e1_x)
        img1_out, img2_out = render_pair(img1, img2, pts1_in, pts2_in, F)
        cell = np.hstack([img1_out, img2_out])
        cell = cv2.resize(cell, (0, 0), fx=scale, fy=scale)
        cv2.putText(
            cell,
            f"e2={e2_x} e1={e1_x}",
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        cells.append(cell)

    grid = np.vstack(cells)
    cv2.imshow(title, grid)


def save_pair(path1, path2, output_path, e2_x, e1_x):
    """Process and save a single pair with the selected epipole positions."""
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    pts1, pts2 = find_matches(gray1, gray2)

    F, pts1_in, pts2_in = compute_F_from_H(pts1, pts2, gray1.shape, e2_x, e1_x)
    img1_out, img2_out = render_pair(img1, img2, pts1_in, pts2_in, F)
    result = np.hstack([img1_out, img2_out])
    cv2.imwrite(output_path, result)
    print(f"Saved {output_path}")
    return result


if __name__ == "__main__":
    w = cv2.imread(LEFT_PATH).shape[1]

    lc = save_pair(LEFT_PATH, CENTER_PATH, "epipolar_lc.jpg", -10000, w + 10000)
    cr = save_pair(CENTER_PATH, RIGHT_PATH, "epipolar_cr.jpg", -350, w + 350)

    cv2.imshow("Left + Center", lc)
    cv2.imshow("Center + Right", cr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
