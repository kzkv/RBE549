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
    base = cv2.GaussianBlur(
        doubled, (0, 0), sigmaX=sigma_diff
    )  # kernel size derived from sigma
    return base


def build_gaussian_octave(base):
    # Produce S+3 progressively blurred images from a single octave base
    gaussians = [base]
    for j in range(1, S + 3):
        sigma_prev = SIGMA_0 * K ** (j - 1)
        sigma_curr = SIGMA_0 * K**j
        sigma_inc = math.sqrt(sigma_curr**2 - sigma_prev**2)
        blurred = cv2.GaussianBlur(gaussians[-1], (0, 0), sigmaX=sigma_inc)
        gaussians.append(blurred)
    return gaussians


def main():
    original = load_image(IMAGE_PATH)

    base = build_base_image(original)
    gaussians = build_gaussian_octave(base)

    sigmas = [SIGMA_0 * K**j for j in range(S + 3)]

    fig, axes = plt.subplots(1, len(gaussians), figsize=(18, 4))
    for j, (g, sigma) in enumerate(zip(gaussians, sigmas)):
        axes[j].imshow(g, cmap="gray")
        axes[j].set_title(f"σ={sigma:.2f}")
        axes[j].axis("off")
    fig.suptitle("Octave 0 — Gaussian Stack")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
