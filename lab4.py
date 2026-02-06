# Tom Kazakov
# RBE 549 Lab 4: Geometric Transformations and Feature Detection

import cv2
import numpy as np

IMAGE_PATH = "UnityHall.png"

ROTATION_ANGLE = 10
SCALE_UP_FACTOR = 1.2


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
    return cv2.warpAffine(img, matrix, (w, h), borderValue=(255, 255, 255))


def scale(img, factor):
    """Scale image by the given factor. Upscale returns full enlarged image; downscale pads with white."""
    if factor == 1.0:
        return img.copy()
    h, w = img.shape[:2]
    new_w, new_h = int(w * factor), int(h * factor)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if factor > 1.0:
        return resized

    canvas = np.full_like(img, 255)
    x_off = (w - new_w) // 2
    y_off = (h - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


if __name__ == "__main__":
    original = load_image()
    rotated = rotate(original, ROTATION_ANGLE)
    scaled_up = scale(original, SCALE_UP_FACTOR)

    cv2.imshow("Original", original)
    cv2.imshow(f"Rotated {ROTATION_ANGLE} deg", rotated)
    cv2.imshow(f"Scaled Up x{SCALE_UP_FACTOR}", scaled_up)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
