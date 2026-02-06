# Tom Kazakov
# RBE 549 Lab 4: Geometric Transformations and Feature Detection

import cv2
import numpy as np

IMAGE_PATH = "UnityHall.png"


def load_image():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Cannot read {IMAGE_PATH}")
    return img


def rotate(img, angle_deg):
    """Rotate image around its center by the given angle."""
    h, w = img.shape[:2]
    center = ((w - 1) / 2.0, (h - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(img, matrix, (w, h))


if __name__ == "__main__":
    original = load_image()
    rotated = rotate(original, 10)

    cv2.imshow("Original", original)
    cv2.imshow("Rotated 10 deg", rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
