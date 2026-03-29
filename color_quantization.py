"""Color quantization using K-Means clustering on nature.png."""

import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = "nature.png"
K_VALUES = [2, 3, 5, 10, 20, 40]
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ATTEMPTS = 10
FLAGS = cv2.KMEANS_RANDOM_CENTERS


def quantize(image, k):
    """Apply K-Means color quantization with k clusters."""
    pixels = image.reshape(-1, 3).astype(np.float32)
    _, labels, centers = cv2.kmeans(pixels, k, None, CRITERIA, ATTEMPTS, FLAGS)
    quantized = centers[labels.flatten()].astype(np.uint8)
    return quantized.reshape(image.shape)


if __name__ == "__main__":
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Cannot load {IMAGE_PATH}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, k in zip(axes.flat, K_VALUES):
        quantized = quantize(image, k)
        quantized_rgb = cv2.cvtColor(quantized, cv2.COLOR_BGR2RGB)
        ax.imshow(quantized_rgb)
        ax.set_title(f"k = {k}")
        ax.axis("off")
        print(f"k={k}: done")

    plt.suptitle("Color Quantization (K-Means)", fontsize=14)
    plt.tight_layout()
    plt.savefig("color_quantization.png", dpi=150)
    plt.show()
