"""Render an AR triangular pyramid on a chessboard using pose estimation."""

from pathlib import Path

import cv2
import numpy as np

# Checkerboard geometry (must match calibration)
BOARD_SIZE = (9, 6)
SQUARE_SIZE_MM = 25.0

# Corner refinement termination criteria
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Persisted calibration data from Lab 8
CAMERA_MATRIX_PATH = "camera_matrix.npy"
DIST_COEFFS_PATH = "dist_coeffs.npy"

# Image to render on
IMAGE_PATH = Path("data/left/left08.jpg")

# Axis length in squares
AXIS_LENGTH = 3

# Pyramid geometry: base inscribed in a 3x3 square region, centered at (4, 2.5)
PYRAMID_CENTER = np.array([4.0, 2.5])
PYRAMID_RADIUS = 1.5  # Circumradius in squares — fits inside a 3x3 area

# Drawing
AXIS_THICKNESS = 3
PYRAMID_COLOR = (255, 0, 255)  # Magenta
PYRAMID_THICKNESS = 3

OUTPUT_AXES = "axes.jpg"
OUTPUT_PYRAMID = "triangular.jpg"


def load_calibration():
    """Load persisted camera matrix and distortion coefficients."""
    mtx = np.load(CAMERA_MATRIX_PATH)
    dist = np.load(DIST_COEFFS_PATH)
    return mtx, dist


def detect_corners(img, board_size=BOARD_SIZE):
    """Find and refine chessboard corners in a single image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, board_size, None)
    if not found:
        raise RuntimeError(f"Chessboard corners not found in {IMAGE_PATH}")
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)


def build_object_points(board_size=BOARD_SIZE):
    """Create 3D object points for the chessboard (z=0 plane)."""
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM
    return objp


def draw_axes(img, rvec, tvec, mtx, dist):
    """Draw RGB axes (X=red, Y=green, Z=blue) at the chessboard origin."""
    length = AXIS_LENGTH * SQUARE_SIZE_MM
    origin = np.float32([[0, 0, 0]])
    axes = np.float32([[length, 0, 0], [0, length, 0], [0, 0, -length]])

    origin_2d, _ = cv2.projectPoints(origin, rvec, tvec, mtx, dist)
    axes_2d, _ = cv2.projectPoints(axes, rvec, tvec, mtx, dist)

    o = tuple(origin_2d.reshape(-1, 2)[0].astype(int))
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # X=red, Y=green, Z=blue
    for pt, color in zip(axes_2d.reshape(-1, 2).astype(int), colors):
        cv2.line(img, o, tuple(pt), color, AXIS_THICKNESS)


def build_pyramid_points():
    """Define a regular tetrahedron in chessboard coordinates."""
    cx, cy = PYRAMID_CENTER * SQUARE_SIZE_MM
    r = PYRAMID_RADIUS * SQUARE_SIZE_MM
    # For a regular tetrahedron, apex height = circumradius * sqrt(2)
    h = -r * np.sqrt(2)

    angles = [np.pi / 2, 7 * np.pi / 6, 11 * np.pi / 6]
    base = np.array(
        [[cx + r * np.cos(a), cy + r * np.sin(a), 0] for a in angles],
        dtype=np.float32,
    )
    apex = np.array([[cx, cy, h]], dtype=np.float32)
    return np.vstack([base, apex])


def draw_pyramid(img, projected):
    """Draw the tetrahedron wireframe in magenta."""
    pts = projected.reshape(-1, 2).astype(int)
    b0, b1, b2, apex = pts[0], pts[1], pts[2], pts[3]

    for start, end in [(b0, b1), (b1, b2), (b2, b0)]:
        cv2.line(img, tuple(start), tuple(end), PYRAMID_COLOR, PYRAMID_THICKNESS)
    for base_pt in [b0, b1, b2]:
        cv2.line(img, tuple(base_pt), tuple(apex), PYRAMID_COLOR, PYRAMID_THICKNESS)


if __name__ == "__main__":
    mtx, dist = load_calibration()
    original = cv2.imread(str(IMAGE_PATH))
    obj_points = build_object_points()
    img_points = detect_corners(original)

    _, rvec, tvec = cv2.solvePnP(obj_points, img_points, mtx, dist)

    # Image 1: RG
    # B axes
    axes_img = original.copy()
    draw_axes(axes_img, rvec, tvec, mtx, dist)
    cv2.imwrite(OUTPUT_AXES, axes_img)
    print(f"Saved {OUTPUT_AXES}")

    # Image 2: Magenta tetrahedron wireframe
    pyramid_img = original.copy()
    pyramid_3d = build_pyramid_points()
    pyramid_2d, _ = cv2.projectPoints(pyramid_3d, rvec, tvec, mtx, dist)
    draw_pyramid(pyramid_img, pyramid_2d)
    cv2.imwrite(OUTPUT_PYRAMID, pyramid_img)
    print(f"Saved {OUTPUT_PYRAMID}")

    cv2.imshow("Axes", axes_img)
    cv2.imshow("Triangular Pyramid", pyramid_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
