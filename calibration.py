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
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    return cv2.undistort(img, mtx, dist, None, new_mtx)


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


if __name__ == "__main__":
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
    cv2.imshow("Original vs Undistorted", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
