# Tom Kazakov
# RBE 549, Week 9 Assignment: Mini Structure-from-Motion pipeline.

import struct
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


FRONTAL_EXTRINSIC = np.array(
    [
        [0.9994459, -0.01668563, 0.02880063, -0.27385663],
        [0.01978507, 0.99363172, -0.11092592, 0.39244263],
        [-0.02676635, 0.11143428, 0.99341128, 0.91339497],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

TOPDOWN_EXTRINSIC = np.array(
    [
        [0.98133378, -0.10538005, -0.16086969, 0.26462035],
        [-0.14785562, 0.12147613, -0.98152038, 2.84310288],
        [0.12297449, 0.9869846, 0.1036276, 3.49775039],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


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


def load_and_undistort(path, K, dist):
    """Load an image and remove lens distortion."""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not load {path}")
    return cv2.undistort(image, K, dist)


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

    h = gray_left.shape[0]
    vis_v = np.vstack([gray_left, gray_right])
    rng = np.random.RandomState(42)
    for m in matches:
        pt_left = tuple(map(int, kp_left[m.queryIdx].pt))
        pt_right_raw = tuple(map(int, kp_right[m.trainIdx].pt))
        pt_right = (pt_right_raw[0], pt_right_raw[1] + h)
        color = tuple(rng.randint(0, 255, 3).tolist())
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

    print(f"  Inliers after RANSAC: {inlier_mask.sum()}/{len(matches)}")
    print(f"  F:\n{F}")

    ones = np.ones((len(pts_left), 1))
    homog_left = np.hstack([pts_left, ones])
    homog_right = np.hstack([pts_right, ones])
    residuals = np.abs(np.sum(homog_right * (homog_left @ F.T), axis=1))
    print(f"  Epipolar constraint residuals:")
    print(f"    Mean: {residuals.mean():.6f}")
    print(f"    Max:  {residuals.max():.6f}")

    return F, pts_left, pts_right


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
    """Triangulate one 3D point via Linear-LS (Hartley & Sturm, Section 5.1)."""
    A = np.array(
        [
            u0[0] * P0[2, :3] - P0[0, :3],
            u0[1] * P0[2, :3] - P0[1, :3],
            u1[0] * P1[2, :3] - P1[0, :3],
            u1[1] * P1[2, :3] - P1[1, :3],
        ]
    )
    b = np.array(
        [
            -(u0[0] * P0[2, 3] - P0[0, 3]),
            -(u0[1] * P0[2, 3] - P0[1, 3]),
            -(u1[0] * P1[2, 3] - P1[0, 3]),
            -(u1[1] * P1[2, 3] - P1[1, 3]),
        ]
    )
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return result


def triangulate_points(pts_left, pts_right, P0, P1):
    """Triangulate all point correspondences."""
    return np.array(
        [LinearLSTriangulation(pl, P0, pr, P1) for pl, pr in zip(pts_left, pts_right)]
    )


def cheirality_check(pts_left, pts_right, P0, P1_candidates, R1, R2, T):
    """Select the P1 candidate that places the most points in front of both cameras."""
    extrinsics = [(R1, T), (R1, -T), (R2, T), (R2, -T)]

    best_idx, best_count, best_points = 0, 0, None
    for i, (P1, (R, t)) in enumerate(zip(P1_candidates, extrinsics)):
        points_3d = triangulate_points(pts_left, pts_right, P0, P1)

        depth_cam0 = points_3d[:, 2]

        homog = np.hstack([points_3d, np.ones((len(points_3d), 1))])
        Rt = np.hstack([R, t])
        depth_cam1 = (Rt @ homog.T)[2]

        in_front = np.sum((depth_cam0 > 0) & (depth_cam1 > 0))
        print(
            f"  Candidate {i + 1}: {in_front}/{len(points_3d)} points in front of both cameras"
        )

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
            rgb_int = (int(r) << 16) | (int(g) << 8) | int(b)
            rgb_packed = struct.unpack("f", struct.pack("I", rgb_int))[0]
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {rgb_packed:.8e}\n")

    print(f"  Saved {n} points to {filename}")


def save_static_views(pcd, width=1280, height=720):
    """Save frontal and top-down screenshots from fixed camera poses."""
    import open3d as o3d

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width, height, width, width, width / 2, height / 2
    )

    for filename, extrinsic in [
        ("view_frontal.png", FRONTAL_EXTRINSIC),
        ("view_topdown.png", TOPDOWN_EXTRINSIC),
    ]:
        cam = o3d.camera.PinholeCameraParameters()
        cam.intrinsic = intrinsic
        cam.extrinsic = extrinsic

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=width, height=height, visible=False)
        vis.add_geometry(pcd)
        vis.get_render_option().point_size = 5.0
        vis.get_render_option().background_color = np.array([0.1, 0.1, 0.1])

        ctl = vis.get_view_control()
        ctl.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)

        for _ in range(30):
            vis.poll_events()
            vis.update_renderer()

        vis.capture_screen_image(filename)
        vis.destroy_window()
        print(f"  Saved {filename}")


def visualize_point_cloud(pcd_path):
    """Save static views and open interactive viewer starting at frontal pose."""
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f"  Loaded {len(pcd.points)} points")

    save_static_views(pcd)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(1280, 720, 1280, 1280, 640, 360)
    cam = o3d.camera.PinholeCameraParameters()
    cam.intrinsic = intrinsic
    cam.extrinsic = FRONTAL_EXTRINSIC

    print("  Opening interactive viewer (Q to quit)...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720, window_name="SfM Point Cloud")
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 5.0
    vis.get_render_option().background_color = np.array([0.1, 0.1, 0.1])

    ctl = vis.get_view_control()
    ctl.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)

    vis.run()
    vis.destroy_window()


def main():
    print("Part 0: Camera Calibration")
    if RECALIBRATE:
        K, dist = calibrate_camera()
    else:
        K, dist = load_calibration()
        print(f"K:\n{K}")

    img_left = load_and_undistort(IMAGE_LEFT, K, dist)
    img_right = load_and_undistort(IMAGE_RIGHT, K, dist)

    print("\nParts 1-2: Feature Extraction & Matching")
    kp_left, kp_right, matches = detect_and_match(img_left, img_right)
    draw_features(img_left, img_right, kp_left, kp_right)
    draw_matches(img_left, img_right, kp_left, kp_right, matches)

    print("\nPart 3: Fundamental Matrix")
    F, pts_left, pts_right = compute_fundamental_matrix(kp_left, kp_right, matches)

    print("\nPart 4: Essential Matrix")
    E = compute_essential_matrix(F, K)

    print("\nPart 5: Decompose Essential Matrix")
    R1, R2, T = decompose_essential(E)

    print("\nParts 6-6.5: Projection Matrices")
    P0, P1_candidates = build_projection_matrices(K, R1, R2, T)

    print("\nParts 7-7.5: Triangulation & Cheirality Check")
    points_3d, best_idx = cheirality_check(
        pts_left, pts_right, P0, P1_candidates, R1, R2, T
    )
    P1 = P1_candidates[best_idx]

    print("\nPart 8: Reprojection Error")
    compute_reprojection_error(points_3d, pts_left, pts_right, P0, P1)

    print("\nPart 9: Save Point Cloud")
    colors = extract_colors(img_left, pts_left)
    SavePCDToFile(points_3d, colors, "point_cloud.pcd")

    print("\nPart 10: Open3D Visualization")
    visualize_point_cloud("point_cloud.pcd")


if __name__ == "__main__":
    main()
