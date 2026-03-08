from pathlib import Path

import cv2
import numpy as np

# Checkerboard geometry
CHECKERBOARD = (9, 6)
SQUARE_SIZE_MM = 25.0

# Corner refinement termination criteria
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Directories
SAMPLE_IMAGE_DIR = Path("data/left")
OWN_IMAGE_DIR = Path("data/checkerboard")
DISTORTED_IMAGE_DIR = Path("data/distorted")

# Ground-truth distortion for Part 2 (k1, k2, p1, p2, k3)
GT_DIST = np.array([0.5, -0.1, 0.01, -0.005, 0.0])


def find_corners(image_dir, board_size=CHECKERBOARD):
    """Detect and refine checkerboard corners in all images from a directory."""
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    obj_points = []
    img_points = []
    images = []
    names = []
    all_images = []
    all_found = []
    image_size = None

    paths = sorted(image_dir.glob("*.jpg"))
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, board_size, None)
        all_images.append(img)
        all_found.append(found)

        if not found:
            print(f"Corners not found: {path.name}")
            continue

        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
        obj_points.append(objp)
        img_points.append(refined)
        images.append(img)
        names.append(path.stem)
        print(f"Corners found: {path.name}")

    print(f"\n{len(images)}/{len(paths)} images usable")
    return obj_points, img_points, images, names, image_size, all_images, all_found


def calibrate(obj_points, img_points, image_size):
    """Run camera calibration and return intrinsics, distortion, and extrinsics."""
    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )
    print(f"RMS reprojection error: {rms:.4f}")
    print(f"\nCamera matrix:\n{mtx}")
    print(f"\nDistortion coefficients:\n{dist}")
    return mtx, dist, rvecs, tvecs


def compute_reprojection_error(obj_points, img_points, rvecs, tvecs, mtx, dist):
    """Compute per-image mean reprojection error using projectPoints."""
    errors = []
    for obj, img, rvec, tvec in zip(obj_points, img_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, mtx, dist)
        error = cv2.norm(img, projected, cv2.NORM_L2) / len(projected)
        errors.append(error)

    print("\nPer-image reprojection error:")
    for i, e in enumerate(errors):
        print(f"  Image {i + 1:2d}: {e:.4f}")
    print(f"  Mean:     {np.mean(errors):.4f}")
    return errors


def undistort_image(img, mtx, dist):
    """Remove lens distortion using the calibrated camera parameters."""
    h, w = img.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    return cv2.undistort(img, mtx, dist, None, new_mtx)


def apply_distortion(img, mtx, dist_coeffs):
    """Apply artificial distortion to an image using the inverse undistortPoints trick."""
    h, w = img.shape[:2]
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    pts = np.stack([u.ravel(), v.ravel()], axis=-1).reshape(-1, 1, 2)
    undistorted_pts = cv2.undistortPoints(pts, mtx, dist_coeffs, P=mtx)
    map_x = undistorted_pts[:, 0, 0].reshape(h, w)
    map_y = undistorted_pts[:, 0, 1].reshape(h, w)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)


def save_calibration(mtx, dist, error):
    """Save camera matrix, distortion coefficients, and reprojection error as .npy files."""
    np.save("camera_matrix.npy", mtx)
    np.save("dist_coeffs.npy", dist)
    np.save("reprojection_error.npy", np.array(error))
    print(
        "\nCalibration saved to camera_matrix.npy, dist_coeffs.npy, reprojection_error.npy"
    )


def build_grid(images, cols):
    """Tile images into a grid, padding the last row if needed."""
    h, w = images[0].shape[:2]
    while len(images) % cols != 0:
        images.append(np.full((h, w, 3), 255, dtype=np.uint8))
    rows = [np.hstack(images[i : i + cols]) for i in range(0, len(images), cols)]
    return np.vstack(rows)


def part1():
    """Part 1: Calibrate OpenCV sample images, undistort, save parameters."""
    obj_pts, img_pts, imgs, img_names, img_size, all_imgs, all_found = find_corners(
        SAMPLE_IMAGE_DIR
    )
    mtx, dist, rvecs, tvecs = calibrate(obj_pts, img_pts, img_size)
    errors = compute_reprojection_error(obj_pts, img_pts, rvecs, tvecs, mtx, dist)
    save_calibration(mtx, dist, np.mean(errors))

    pairs = []
    for img, name, error in zip(imgs, img_names, errors):
        undistorted = undistort_image(img, mtx, dist)
        cv2.putText(img, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(
            undistorted,
            f"err: {error:.4f}px",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        pairs.append(np.hstack([img, undistorted]))

    grid = build_grid(pairs, cols=2)
    cv2.imshow("Part 1: Original vs Undistorted", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def part2_baseline():
    """Part 2: Calibrate own clean images, undistort, and show grid."""
    obj_pts, img_pts, imgs, names, img_size, _, _ = find_corners(OWN_IMAGE_DIR)
    mtx, dist, rvecs, tvecs = calibrate(obj_pts, img_pts, img_size)
    errors = compute_reprojection_error(obj_pts, img_pts, rvecs, tvecs, mtx, dist)

    pairs = []
    for img, name, error in zip(imgs, names, errors):
        undistorted = undistort_image(img, mtx, dist)
        h, w = img.shape[:2]
        scale = 640 / w
        img_small = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        und_small = cv2.resize(undistorted, (0, 0), fx=scale, fy=scale)
        cv2.putText(
            img_small, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.putText(
            und_small,
            f"err: {error:.4f}px",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        pairs.append(np.hstack([img_small, und_small]))

    grid = build_grid(pairs, cols=2)
    cv2.imshow("Part 2: Baseline Original vs Undistorted", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def part2_distort():
    """Part 2: Apply artificial distortion to own images and save them."""
    obj_pts, img_pts, imgs, names, img_size, _, _ = find_corners(OWN_IMAGE_DIR)
    mtx, dist, rvecs, tvecs = calibrate(obj_pts, img_pts, img_size)

    print(f"\nApplying ground-truth distortion: {GT_DIST}")
    DISTORTED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for img, name in zip(imgs, names):
        distorted = apply_distortion(img, mtx, GT_DIST)
        path = DISTORTED_IMAGE_DIR / f"{name}.jpg"
        cv2.imwrite(str(path), distorted)
        print(f"Saved {path}")


def part2_calibrate_distorted():
    """Part 2: Calibrate distorted images, compare recovered params to ground truth."""
    print("\n--- Baseline calibration (clean images) ---")
    obj_pts_b, img_pts_b, _, _, img_size_b, _, _ = find_corners(OWN_IMAGE_DIR)
    _, baseline_dist, _, _ = calibrate(obj_pts_b, img_pts_b, img_size_b)

    print("\n--- Distorted image calibration ---")
    obj_pts, img_pts, imgs, names, img_size, _, _ = find_corners(DISTORTED_IMAGE_DIR)
    mtx, dist, rvecs, tvecs = calibrate(obj_pts, img_pts, img_size)
    compute_reprojection_error(obj_pts, img_pts, rvecs, tvecs, mtx, dist)

    labels = ["k1", "k2", "p1", "p2", "k3"]
    baseline = baseline_dist.ravel()
    recovered = dist.ravel()
    expected = baseline + GT_DIST

    print("\n--- Parameter Comparison ---")
    print(
        f"{'Param':>6s} {'Baseline':>12s} {'GT Applied':>12s} "
        f"{'Expected':>12s} {'Recovered':>12s} {'Delta':>12s}"
    )
    for label, b, g, e, r in zip(labels, baseline, GT_DIST, expected, recovered):
        print(
            f"{label:>6s} {b:>12.6f} {g:>12.6f} "
            f"{e:>12.6f} {r:>12.6f} {r - e:>12.6f}"
        )


SHOWCASE_COUNT = 6


def part2_final_visuals():
    """Part 2: Three-panel comparison of clean, distorted, and undistorted images."""
    obj_pts_c, img_pts_c, clean_imgs, clean_names, _, _, _ = find_corners(OWN_IMAGE_DIR)
    mtx_c, dist_c, _, _ = calibrate(obj_pts_c, img_pts_c, clean_imgs[0].shape[1::-1])

    obj_pts_d, img_pts_d, dist_imgs, dist_names, img_size_d, _, _ = find_corners(
        DISTORTED_IMAGE_DIR
    )
    mtx_d, dist_d, _, _ = calibrate(obj_pts_d, img_pts_d, img_size_d)

    clean_by_name = dict(zip(clean_names, clean_imgs))
    dist_by_name = dict(zip(dist_names, dist_imgs))

    shared_names = [n for n in clean_names if n in dist_by_name][:SHOWCASE_COUNT]

    panels = []
    h, w = clean_imgs[0].shape[:2]
    scale = 480 / w
    for name in shared_names:
        clean = cv2.resize(clean_by_name[name], (0, 0), fx=scale, fy=scale)
        distorted = cv2.resize(dist_by_name[name], (0, 0), fx=scale, fy=scale)
        undistorted = cv2.resize(
            undistort_image(dist_by_name[name], mtx_d, dist_d),
            (0, 0),
            fx=scale,
            fy=scale,
        )
        cv2.putText(
            clean, "Original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        cv2.putText(
            distorted,
            "Distorted",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            undistorted,
            "Undistorted",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )
        panels.append(np.hstack([clean, distorted, undistorted]))

    grid = np.vstack(panels)
    cv2.imshow("Part 2: Original / Distorted / Undistorted", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


RUN_PART1 = False
RUN_PART2_BASELINE = False
RUN_PART2_DISTORT = False
RUN_PART2_CALIBRATE = False
RUN_PART2_VISUALS = True

if __name__ == "__main__":
    if RUN_PART1:
        part1()
    if RUN_PART2_BASELINE:
        part2_baseline()
    if RUN_PART2_DISTORT:
        part2_distort()
    if RUN_PART2_CALIBRATE:
        part2_calibrate_distorted()
    if RUN_PART2_VISUALS:
        part2_final_visuals()
