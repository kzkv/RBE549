"""Compute and visualize epipolar lines and epipoles for stereo image pairs."""

import cv2
import numpy as np

# Input images
LEFT_PATH = "globe_left.jpg"
CENTER_PATH = "globe_center.jpg"
RIGHT_PATH = "globe_right.jpg"

# SIFT matching
RATIO_THRESHOLD = 0.75
RANSAC_CONFIDENCE = 0.999
RANSAC_THRESHOLD = 1.0

# Spatial bucketing for geographically distributed matches
GRID_COLS = 10
GRID_ROWS = 8
MAX_PER_CELL = 3

# Visualization
NUM_LINES = 20
LINE_THICKNESS = 1
POINT_RADIUS = 5
EPIPOLE_RADIUS = 10
EPIPOLE_COLOR = (255, 255, 255)


def find_matches(img1, img2):
    """Detect SIFT features and return spatially distributed ratio-tested matches."""
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    raw_matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in raw_matches:
        if m.distance < RATIO_THRESHOLD * n.distance:
            good.append(m)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    pts1, pts2 = bucket_matches(pts1, pts2, img1.shape)
    return pts1, pts2


def bucket_matches(pts1, pts2, img_shape):
    """Select the best match per grid cell to ensure spatial distribution."""
    h, w = img_shape[:2]
    cell_h = h / GRID_ROWS
    cell_w = w / GRID_COLS

    selected = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            x_lo, x_hi = c * cell_w, (c + 1) * cell_w
            y_lo, y_hi = r * cell_h, (r + 1) * cell_h
            mask = (
                (pts1[:, 0] >= x_lo)
                & (pts1[:, 0] < x_hi)
                & (pts1[:, 1] >= y_lo)
                & (pts1[:, 1] < y_hi)
            )
            indices = np.where(mask)[0]
            if len(indices) > 0:
                selected.extend(indices[:MAX_PER_CELL])

    selected = np.array(selected)
    return pts1[selected], pts2[selected]


def compute_fundamental(pts1, pts2):
    """Compute the fundamental matrix using RANSAC and return inlier points."""
    F, mask = cv2.findFundamentalMat(
        pts1, pts2, cv2.FM_RANSAC, RANSAC_THRESHOLD, RANSAC_CONFIDENCE
    )
    inliers = mask.ravel().astype(bool)
    return F, pts1[inliers], pts2[inliers]


def compute_epipole(F):
    """Compute the epipole as the null space of F (right null vector via SVD)."""
    _, _, Vt = np.linalg.svd(F)
    e = Vt[-1]
    return (e / e[2])[:2].astype(int)


def draw_epilines(img1, img2, pts1, pts2, F):
    """Draw epipolar lines on img1 for points in img2, and mark corresponding points."""
    img1_out = img1.copy()
    img2_out = img2.copy()
    h, w = img1.shape[:2]

    # Subsample for clarity
    rng = np.random.default_rng(42)
    indices = rng.choice(len(pts1), size=min(NUM_LINES, len(pts1)), replace=False)
    pts1_sub = pts1[indices]
    pts2_sub = pts2[indices]

    # Epipolar lines in img1 for points in img2
    lines1 = cv2.computeCorrespondEpilines(pts2_sub.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)

    # Epipolar lines in img2 for points in img1
    lines2 = cv2.computeCorrespondEpilines(pts1_sub.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)

    for line, pt1, pt2 in zip(lines1, pts1_sub, pts2_sub):
        color = tuple(rng.integers(0, 255, size=3).tolist())
        a, b, c = line
        x0, x1 = 0, w
        y0 = int(-c / b)
        y1 = int(-(c + a * w) / b)
        cv2.line(img1_out, (x0, y0), (x1, y1), color, LINE_THICKNESS)
        cv2.circle(img1_out, tuple(pt1.astype(int)), POINT_RADIUS, color, -1)
        cv2.circle(img2_out, tuple(pt2.astype(int)), POINT_RADIUS, color, -1)

    for line, pt1, pt2 in zip(lines2, pts1_sub, pts2_sub):
        color = tuple(rng.integers(0, 255, size=3).tolist())
        a, b, c = line
        x0, x1 = 0, w
        y0 = int(-c / b)
        y1 = int(-(c + a * w) / b)
        cv2.line(img2_out, (x0, y0), (x1, y1), color, LINE_THICKNESS)

    # Draw epipoles
    e1 = compute_epipole(F.T)
    e2 = compute_epipole(F)
    cv2.circle(img1_out, tuple(e1), EPIPOLE_RADIUS, EPIPOLE_COLOR, -1)
    cv2.circle(img2_out, tuple(e2), EPIPOLE_RADIUS, EPIPOLE_COLOR, -1)

    return img1_out, img2_out


def process_pair(path1, path2, output_path):
    """Run the full epipolar pipeline for one image pair."""
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    pts1, pts2 = find_matches(gray1, gray2)
    print(f"{path1} + {path2}: {len(pts1)} matches after ratio test")

    F, pts1_in, pts2_in = compute_fundamental(pts1, pts2)
    print(f"  {len(pts1_in)} inliers after RANSAC")
    print(f"  F =\n{F}")
    e1 = compute_epipole(F.T)
    e2 = compute_epipole(F)
    print(f"  Epipole in img1: {e1}")
    print(f"  Epipole in img2: {e2}")

    img1_out, img2_out = draw_epilines(img1, img2, pts1_in, pts2_in, F)
    result = np.hstack([img1_out, img2_out])

    cv2.imwrite(output_path, result)
    print(f"  Saved {output_path}")
    return result


if __name__ == "__main__":
    lc = process_pair(LEFT_PATH, CENTER_PATH, "epipolar_lc.jpg")
    cr = process_pair(CENTER_PATH, RIGHT_PATH, "epipolar_cr.jpg")

    cv2.imshow("Left + Center", lc)
    cv2.imshow("Center + Right", cr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
