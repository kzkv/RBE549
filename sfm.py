"""Mini Structure-from-Motion pipeline."""

from pathlib import Path

import cv2
import numpy as np

IMAGE_LEFT = "capture_left.jpg"
IMAGE_RIGHT = "capture_right.jpg"
CALIBRATION_MATRIX = "camera_matrix.npy"
DISTORTION_COEFFS = "dist_coeffs.npy"
CHECKERBOARD_DIR = Path("data/checkerboard")
CHECKERBOARD = (9, 6)
SQUARE_SIZE_MM = 25.0
CORNER_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

RECALIBRATE = False
RATIO_THRESHOLD = 0.5


def calibrate_camera():
    """Calibrate from checkerboard images, save K and dist, return them."""
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    obj_points, img_points = [], []
    image_size = None

    paths = sorted(CHECKERBOARD_DIR.glob("*.jpg"))
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        if not found:
            print(f"  Corners not found: {path.name}")
            continue

        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CORNER_CRITERIA)
        obj_points.append(objp)
        img_points.append(refined)
        print(f"  Corners found: {path.name}")

    print(f"\n  {len(img_points)}/{len(paths)} images usable")
    print(f"  Image size: {image_size[0]}x{image_size[1]}")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    # Per-image reprojection error
    errors = []
    for obj, img, rvec, tvec in zip(obj_points, img_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        errors.append(cv2.norm(img, projected, cv2.NORM_L2) / len(projected))

    print(f"\n  RMS reprojection error: {rms:.4f} px")
    print(f"  Mean per-image error:   {np.mean(errors):.4f} px")
    print(f"\n  Camera matrix K:\n{K}")
    print(f"\n  Distortion coefficients:\n{dist}")

    np.save(CALIBRATION_MATRIX, K)
    np.save(DISTORTION_COEFFS, dist)
    print(f"\n  Saved to {CALIBRATION_MATRIX}, {DISTORTION_COEFFS}")
    return K, dist


def load_calibration():
    """Load intrinsic matrix and distortion coefficients from disk."""
    K = np.load(CALIBRATION_MATRIX)
    dist = np.load(DISTORTION_COEFFS)
    return K, dist


def detect_and_match(img_left, img_right):
    """Detect SIFT features in both images and match with ratio test."""
    sift = cv2.SIFT_create()
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    kp_left, des_left = sift.detectAndCompute(gray_left, None)
    kp_right, des_right = sift.detectAndCompute(gray_right, None)
    print(f"  Keypoints: left={len(kp_left)}, right={len(kp_right)}")

    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(des_left, des_right, k=2)

    good_matches = []
    for m, n in raw_matches:
        if m.distance < RATIO_THRESHOLD * n.distance:
            good_matches.append(m)

    print(
        f"  Matches after ratio test (threshold={RATIO_THRESHOLD}): {len(good_matches)}"
    )
    return kp_left, kp_right, good_matches


def to_grayscale_bgr(img):
    """Convert to grayscale and back to BGR so color markers are visible."""
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def draw_features(img_left, img_right, kp_left, kp_right):
    """Draw detected keypoints over grayscale images and save vertically."""
    gray_left = to_grayscale_bgr(img_left)
    gray_right = to_grayscale_bgr(img_right)
    vis_left = cv2.drawKeypoints(
        gray_left,
        kp_left,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    vis_right = cv2.drawKeypoints(
        gray_right,
        kp_right,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    combined = np.vstack([vis_left, vis_right])
    cv2.imwrite("features.jpg", combined)
    print("  Saved features.jpg")


def draw_matches(img_left, img_right, kp_left, kp_right, matches):
    """Draw matches in both horizontal and vertical layouts."""
    gray_left = to_grayscale_bgr(img_left)
    gray_right = to_grayscale_bgr(img_right)

    # Horizontal: drawMatches side by side
    vis_h = cv2.drawMatches(
        gray_left,
        kp_left,
        gray_right,
        kp_right,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite("matches_horizontal.jpg", vis_h)

    # Vertical: manual drawing on stacked images
    h, w = gray_left.shape[:2]
    vis_v = np.vstack([gray_left, gray_right])
    for m in matches:
        pt_left = tuple(map(int, kp_left[m.queryIdx].pt))
        pt_right_raw = tuple(map(int, kp_right[m.trainIdx].pt))
        pt_right = (pt_right_raw[0], pt_right_raw[1] + h)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.line(vis_v, pt_left, pt_right, color, 1)
        cv2.circle(vis_v, pt_left, 3, color, -1)
        cv2.circle(vis_v, pt_right, 3, color, -1)
    cv2.imwrite("matches_vertical.jpg", vis_v)

    print(
        f"  Saved matches_horizontal.jpg, matches_vertical.jpg ({len(matches)} matches)"
    )


def compute_fundamental_matrix(kp_left, kp_right, matches):
    """Compute F via RANSAC, filter to inliers, verify epipolar constraint."""
    pts_left = np.float32([kp_left[m.queryIdx].pt for m in matches])
    pts_right = np.float32([kp_right[m.trainIdx].pt for m in matches])

    F, mask = cv2.findFundamentalMat(pts_left, pts_right, cv2.FM_RANSAC)
    inlier_mask = mask.ravel().astype(bool)
    pts_left = pts_left[inlier_mask]
    pts_right = pts_right[inlier_mask]
    inlier_matches = [m for m, flag in zip(matches, inlier_mask) if flag]

    print(f"  Inliers after RANSAC: {len(inlier_matches)}/{len(matches)}")
    print(f"  F:\n{F}")

    # Verify epipolar constraint: q_R^T @ F @ q_L = 0
    ones = np.ones((len(pts_left), 1))
    homog_left = np.hstack([pts_left, ones])
    homog_right = np.hstack([pts_right, ones])
    residuals = np.abs(np.sum(homog_right * (homog_left @ F.T), axis=1))
    print(f"  Epipolar constraint residuals:")
    print(f"    Mean: {residuals.mean():.6f}")
    print(f"    Max:  {residuals.max():.6f}")

    return F, pts_left, pts_right, inlier_matches


def compute_essential_matrix(F, K):
    """Compute E from F and K, verify det(E) = 0."""
    E = K.T @ F @ K
    print(f"  E:\n{E}")
    print(f"  det(E) = {np.linalg.det(E):.6e}")
    return E


def decompose_essential(E):
    """Decompose E into two candidate rotations and one translation."""
    R1, R2, T = cv2.decomposeEssentialMat(E)
    print(f"  R1:\n{R1}")
    print(f"  R2:\n{R2}")
    print(f"  T:\n{T.ravel()}")
    return R1, R2, T


def build_projection_matrices(K, R1, R2, T):
    """Form P0 and four candidate P1 matrices."""
    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    candidates = [
        K @ np.hstack([R1, T]),
        K @ np.hstack([R1, -T]),
        K @ np.hstack([R2, T]),
        K @ np.hstack([R2, -T]),
    ]
    print(f"  P0:\n{P0}")
    for i, P1 in enumerate(candidates):
        print(f"  P1 candidate {i + 1}:\n{P1}")
    return P0, candidates


def LinearLSTriangulation(u0, P0, u1, P1):
    """Triangulate a single 3D point from two views using the Linear-LS method (Hartley & Sturm, Section 5.1)."""
    # Each view contributes two equations: (u * p3^T - p1^T) @ X = 0
    # With X = (X, Y, Z, 1)^T, move the 4th column to RHS → A @ [X,Y,Z] = b
    A = np.array([
        u0[0] * P0[2, :3] - P0[0, :3],
        u0[1] * P0[2, :3] - P0[1, :3],
        u1[0] * P1[2, :3] - P1[0, :3],
        u1[1] * P1[2, :3] - P1[1, :3],
    ])
    b = np.array([
        -(u0[0] * P0[2, 3] - P0[0, 3]),
        -(u0[1] * P0[2, 3] - P0[1, 3]),
        -(u1[0] * P1[2, 3] - P1[0, 3]),
        -(u1[1] * P1[2, 3] - P1[1, 3]),
    ])
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return result


def triangulate_points(pts_left, pts_right, P0, P1):
    """Triangulate all point correspondences."""
    points_3d = np.array([
        LinearLSTriangulation(pl, P0, pr, P1)
        for pl, pr in zip(pts_left, pts_right)
    ])
    return points_3d


def cheirality_check(pts_left, pts_right, P0, P1_candidates, R1, R2, T):
    """Triangulate with each P1 candidate, select the one with most points in front of both cameras."""
    extrinsics = [
        (R1, T), (R1, -T), (R2, T), (R2, -T),
    ]

    best_idx, best_count, best_points = 0, 0, None
    for i, (P1, (R, t)) in enumerate(zip(P1_candidates, extrinsics)):
        points_3d = triangulate_points(pts_left, pts_right, P0, P1)

        # Depth in camera 0: Z > 0
        depth_cam0 = points_3d[:, 2]

        # Depth in camera 1: third row of R @ X + t
        homog = np.hstack([points_3d, np.ones((len(points_3d), 1))])
        Rt = np.hstack([R, t])
        depth_cam1 = (Rt @ homog.T)[2]

        in_front = np.sum((depth_cam0 > 0) & (depth_cam1 > 0))
        print(f"  Candidate {i + 1}: {in_front}/{len(points_3d)} points in front of both cameras")

        if in_front > best_count:
            best_idx, best_count, best_points = i, in_front, points_3d

    print(f"  Selected candidate {best_idx + 1}")
    return best_points, best_idx


def compute_reprojection_error(points_3d, pts_left, pts_right, P0, P1):
    """Compute mean reprojection error for both cameras."""
    homog = np.hstack([points_3d, np.ones((len(points_3d), 1))])

    proj0 = (P0 @ homog.T).T
    proj0 = proj0[:, :2] / proj0[:, 2:3]
    errors0 = np.linalg.norm(pts_left - proj0, axis=1)

    proj1 = (P1 @ homog.T).T
    proj1 = proj1[:, :2] / proj1[:, 2:3]
    errors1 = np.linalg.norm(pts_right - proj1, axis=1)

    print(f"  Camera 0: mean={errors0.mean():.4f} px, max={errors0.max():.4f} px")
    print(f"  Camera 1: mean={errors1.mean():.4f} px, max={errors1.max():.4f} px")
    print(f"  Overall:  mean={(errors0.mean() + errors1.mean()) / 2:.4f} px")
    return errors0, errors1


def extract_colors(img, pts):
    """Sample BGR pixel color at each feature location, return as RGB."""
    colors = []
    h, w = img.shape[:2]
    for pt in pts:
        x, y = int(round(pt[0])), int(round(pt[1]))
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)
        b, g, r = img[y, x]
        colors.append((r, g, b))
    return np.array(colors, dtype=np.uint8)


def SavePCDToFile(points_3d, colors, filename):
    """Save 3D points with RGB color to an ASCII PCD file."""
    import struct

    n = len(points_3d)
    header = (
        f"VERSION .7\n"
        f"FIELDS x y z rgb\n"
        f"SIZE 4 4 4 4\n"
        f"TYPE F F F F\n"
        f"COUNT 1 1 1 1\n"
        f"WIDTH {n}\n"
        f"HEIGHT 1\n"
        f"VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        f"DATA ascii\n"
    )
    with open(filename, "w") as f:
        f.write(header)
        for pt, (r, g, b) in zip(points_3d, colors):
            rgb_packed = struct.unpack("f", struct.pack("I", (int(r) << 16) | (int(g) << 8) | int(b)))[0]
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {rgb_packed:.8e}\n")

    print(f"  Saved {n} points to {filename}")


def load_and_undistort(path, K, dist):
    """Load an image and remove lens distortion."""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not load {path}")
    return cv2.undistort(image, K, dist)


def main():
    # Part 0: Camera calibration
    print("=" * 60)
    print("PART 0: Camera Calibration")
    print("=" * 60)
    if RECALIBRATE:
        K, dist = calibrate_camera()
    else:
        K, dist = load_calibration()
        print(f"  Loaded K from {CALIBRATION_MATRIX}")
        print(f"  K:\n{K}")

    # Load and undistort stereo pair
    print(f"\nUndistorting {IMAGE_LEFT} and {IMAGE_RIGHT}...")
    img_left = load_and_undistort(IMAGE_LEFT, K, dist)
    img_right = load_and_undistort(IMAGE_RIGHT, K, dist)
    print(f"  Left image:  {img_left.shape}")
    print(f"  Right image: {img_right.shape}")

    # Parts 1-2: Feature extraction and matching
    print("\n" + "=" * 60)
    print("PARTS 1-2: Feature Extraction & Matching")
    print("=" * 60)
    kp_left, kp_right, matches = detect_and_match(img_left, img_right)
    draw_features(img_left, img_right, kp_left, kp_right)
    draw_matches(img_left, img_right, kp_left, kp_right, matches)

    # Part 3: Fundamental matrix
    print("\n" + "=" * 60)
    print("PART 3: Fundamental Matrix")
    print("=" * 60)
    F, pts_left, pts_right, inlier_matches = compute_fundamental_matrix(
        kp_left, kp_right, matches
    )

    # Part 4: Essential matrix
    print("\n" + "=" * 60)
    print("PART 4: Essential Matrix")
    print("=" * 60)
    E = compute_essential_matrix(F, K)

    # Part 5: Recover R and T
    print("\n" + "=" * 60)
    print("PART 5: Decompose Essential Matrix")
    print("=" * 60)
    R1, R2, T = decompose_essential(E)

    # Parts 6-6.5: Projection matrices
    print("\n" + "=" * 60)
    print("PARTS 6-6.5: Projection Matrices")
    print("=" * 60)
    P0, P1_candidates = build_projection_matrices(K, R1, R2, T)

    # Parts 7-7.5: Triangulation and cheirality check
    print("\n" + "=" * 60)
    print("PARTS 7-7.5: Triangulation & Cheirality Check")
    print("=" * 60)
    points_3d, best_idx = cheirality_check(
        pts_left, pts_right, P0, P1_candidates, R1, R2, T
    )
    P1 = P1_candidates[best_idx]

    # Part 8: Reprojection error
    print("\n" + "=" * 60)
    print("PART 8: Reprojection Error")
    print("=" * 60)
    errors0, errors1 = compute_reprojection_error(
        points_3d, pts_left, pts_right, P0, P1
    )

    # Part 9: Save colored point cloud
    print("\n" + "=" * 60)
    print("PART 9: Save Point Cloud")
    print("=" * 60)
    colors = extract_colors(img_left, pts_left)
    SavePCDToFile(points_3d, colors, "point_cloud.pcd")


if __name__ == "__main__":
    main()
