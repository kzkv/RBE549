# Tom Kazakov
# RBE 549 Week 4: SIFT Scale-Space Extrema Detection and Keypoint Localization
# Reference: Lowe, "Distinctive Image Features from Scale-Invariant Keypoints," IJCV 2004

import math

import cv2
import matplotlib.pyplot as plt
import numpy as np

IMAGE_PATH = "Fabio.png"

# Constants extracted from the paper
SIGMA_0 = 1.6
ASSUMED_BLUR = 0.5
S = 3
NUM_OCTAVES = 4
K = 2 ** (1 / S)
CONTRAST_THRESHOLD = 0.03
EDGE_RATIO = 10
MAX_INTERP_STEPS = 5


def load_image(path):
    img = cv2.imread(
        path, cv2.IMREAD_GRAYSCALE
    )  # SIFT operates on luminance and doesn't need color
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img.astype(np.float64) / 255.0


def build_base_image(img):
    # Double the image and pre-blur
    doubled = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    sigma_diff = math.sqrt(SIGMA_0**2 - (2 * ASSUMED_BLUR) ** 2)
    ksize = int(2 * math.ceil(3 * sigma_diff) + 1)
    base = cv2.GaussianBlur(doubled, (ksize, ksize), sigmaX=sigma_diff)
    return base


def main():
    original = load_image(IMAGE_PATH)
    print(f"Loaded {IMAGE_PATH}: shape={original.shape}, dtype={original.dtype}")

    base = build_base_image(original)
    print(f"Base image: shape={base.shape} (2x upsample + pre-blur to sigma={SIGMA_0})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title(f"Original ({original.shape[1]}x{original.shape[0]})")
    axes[0].axis("off")
    axes[1].imshow(base, cmap="gray")
    axes[1].set_title(f"Base ({base.shape[1]}x{base.shape[0]})")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
