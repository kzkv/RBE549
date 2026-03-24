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


def load_and_undistort(path, K, dist):
    """Load an image and remove lens distortion."""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not load {path}")
    return cv2.undistort(image, K, dist)


def main():
    # Part 0: Calibrate camera from checkerboard images
    print("=" * 60)
    print("PART 0: Camera Calibration")
    print("=" * 60)
    K, dist = calibrate_camera()

    # Load and undistort stereo pair
    print(f"\nUndistorting {IMAGE_LEFT} and {IMAGE_RIGHT}...")
    img_left = load_and_undistort(IMAGE_LEFT, K, dist)
    img_right = load_and_undistort(IMAGE_RIGHT, K, dist)
    print(f"  Left image:  {img_left.shape}")
    print(f"  Right image: {img_right.shape}")


if __name__ == "__main__":
    main()
