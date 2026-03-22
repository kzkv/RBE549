"""Compute a disparity map from a stereo image pair."""

import cv2
import numpy as np

# Input stereo pair
LEFT_PATH = "aloeL.jpg"
RIGHT_PATH = "aloeR.jpg"

# StereoSGBM parameters (used for the saved output)
MIN_DISPARITY = 0
NUM_DISPARITIES = 128  # Must be divisible by 16
BLOCK_SIZE = 15

# Parameter sweeps — 3x3 grids, holding the other param at the saved value
DISP_SWEEP = [64, 80, 96, 112, 128, 144, 160, 176, 192]
BLOCK_SWEEP = [3, 5, 7, 9, 11, 13, 15, 17, 19]

OUTPUT_PATH = "displarity.jpg"


def compute_disparity(left_gray, right_gray, num_disp, block_size):
    """Compute normalized disparity map using Semi-Global Block Matching."""
    stereo = cv2.StereoSGBM_create(
        minDisparity=MIN_DISPARITY,
        numDisparities=num_disp,
        blockSize=block_size,
    )
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    return cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def build_sweep_grid(left, right, params, make_label):
    """Build a 3x3 grid of disparity maps from a list of (numDisp, blockSize) pairs."""
    h, w = left.shape[:2]
    scale = 400 / w
    cells = []
    for nd, bs in params:
        cell = compute_disparity(left, right, nd, bs)
        cell = cv2.resize(cell, (0, 0), fx=scale, fy=scale)
        cell = cv2.cvtColor(cell, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            cell,
            make_label(nd, bs),
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        cells.append(cell)
    rows = [np.hstack(cells[i : i + 3]) for i in range(0, 9, 3)]
    return np.vstack(rows)


if __name__ == "__main__":
    left = cv2.imread(LEFT_PATH, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(RIGHT_PATH, cv2.IMREAD_GRAYSCALE)

    # Saved output with the selected parameters
    disparity = compute_disparity(left, right, NUM_DISPARITIES, BLOCK_SIZE)
    label = f"SGBM  numDisp={NUM_DISPARITIES}  block={BLOCK_SIZE}"
    cv2.putText(disparity, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
    cv2.imwrite(OUTPUT_PATH, disparity)
    print(f"Saved {OUTPUT_PATH}")

    disp_grid = build_sweep_grid(
        left,
        right,
        [(nd, BLOCK_SIZE) for nd in DISP_SWEEP],
        lambda nd, bs: f"nD={nd}  blk={bs}",
    )
    block_grid = build_sweep_grid(
        left,
        right,
        [(NUM_DISPARITIES, bs) for bs in BLOCK_SWEEP],
        lambda nd, bs: f"nD={nd}  blk={bs}",
    )

    cv2.imshow("Disparity Range Sweep", disp_grid)
    cv2.imshow("Block Size Sweep", block_grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
