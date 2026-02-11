# Tom Kazakov
# RBE 549 Week 4: SIFT Scale-Space Extrema Detection and Keypoint Localization
# Reference: Lowe, "Distinctive Image Features from Scale-Invariant Keypoints," IJCV 2004

import math

import cv2
import matplotlib.pyplot as plt
import numpy as np

IMAGE_PATH = "Fabio.png"
COMPARE_OPENCV = True

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


def build_dog_octave(gaussians):
    # DoG, difference of adjacent Gaussian images; approximates Laplacian of Gaussian
    return [gaussians[j + 1] - gaussians[j] for j in range(len(gaussians) - 1)]


def build_gaussian_pyramid(base, num_octaves):
    pyramid = []
    for o in range(num_octaves):
        gaussians = build_gaussian_octave(base)
        pyramid.append(gaussians)
        # Downsample the image at 2*sigma_0 to become the next octave's base
        base = cv2.resize(gaussians[S], None, fx=0.5, fy=0.5,
                          interpolation=cv2.INTER_NEAREST)
    return pyramid


def build_dog_pyramid(gaussian_pyramid):
    return [build_dog_octave(g) for g in gaussian_pyramid]


def find_extrema(dog_pyr):
    # Compare each pixel to its 26 neighbors in the 3x3x3 scale-space cube
    keypoints = []
    for o, dogs in enumerate(dog_pyr):
        for j in range(1, len(dogs) - 1):
            below, current, above = dogs[j - 1], dogs[j], dogs[j + 1]
            h, w = current.shape
            center = current[1:-1, 1:-1]

            is_max = np.ones(center.shape, dtype=bool)
            is_min = np.ones(center.shape, dtype=bool)

            for layer in (below, current, above):
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if layer is current and dr == 0 and dc == 0:
                            continue
                        neighbor = layer[1 + dr:h - 1 + dr, 1 + dc:w - 1 + dc]
                        is_max &= center > neighbor
                        is_min &= center < neighbor

            rows, cols = np.where(is_max | is_min)
            for r, c in zip(rows, cols):
                keypoints.append((o, j, r + 1, c + 1))

    return keypoints


def main():
    original = load_image(IMAGE_PATH)

    base = build_base_image(original)
    gauss_pyr = build_gaussian_pyramid(base, NUM_OCTAVES)
    dog_pyr = build_dog_pyramid(gauss_pyr)

    keypoints = find_extrema(dog_pyr)

    per_octave = [sum(1 for kp in keypoints if kp[0] == o) for o in range(NUM_OCTAVES)]
    print(f"Raw extrema: {len(keypoints)} total, per octave: {per_octave}")

    # Map keypoint coordinates to original image space
    scale = lambda o: 2 ** (o - 1)
    kp_x = [kp[3] * scale(kp[0]) for kp in keypoints]
    kp_y = [kp[2] * scale(kp[0]) for kp in keypoints]

    if COMPARE_OPENCV:
        gray_u8 = (original * 255).astype(np.uint8)
        sift_all = cv2.SIFT_create(contrastThreshold=0.0, edgeThreshold=1000)
        sift_default = cv2.SIFT_create()
        kps_all = sift_all.detect(gray_u8)
        kps_default = sift_default.detect(gray_u8)
        print(f"OpenCV SIFT (filters off): {len(kps_all)}")
        print(f"OpenCV SIFT (defaults):    {len(kps_default)}")

        cv_x = [kp.pt[0] for kp in kps_all]
        cv_y = [kp.pt[1] for kp in kps_all]

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(original, cmap="gray")
        axes[0].plot(kp_x, kp_y, "r+", markersize=3)
        axes[0].set_title(f"Ours (N={len(keypoints)})")
        axes[0].axis("off")
        axes[1].imshow(original, cmap="gray")
        axes[1].plot(cv_x, cv_y, "b+", markersize=3)
        axes[1].set_title(f"OpenCV filters off (N={len(kps_all)})")
        axes[1].axis("off")
        fig.suptitle("Raw Extrema Comparison")
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=(8, 8))
        plt.imshow(original, cmap="gray")
        plt.plot(kp_x, kp_y, "r+", markersize=3)
        plt.title(f"Raw Extrema (N={len(keypoints)})")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
