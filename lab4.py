# Tom Kazakov
# RBE 549 Lab 4: Geometric Transformations and Feature Detection

import cv2
import numpy as np

IMAGE_PATH = "UnityHall.png"

ROTATION_ANGLE = 10
SCALE_UP_FACTOR = 1.2
SCALE_DOWN_FACTOR = 0.8

GRID_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
GRID_LABEL_SCALE = 0.7
GRID_LABEL_THICKNESS = 2
GRID_LABEL_MARGIN = 80
GRID_LABEL_PAD = 30


def load_image():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Cannot read {IMAGE_PATH}")
    return img


def rotate(img, angle_deg):
    """Rotate image around its center, expanding the canvas to fit the full result."""
    h, w = img.shape[:2]
    center = ((w - 1) / 2.0, (h - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    matrix[0, 2] += (new_w - w) / 2.0
    matrix[1, 2] += (new_h - h) / 2.0

    return cv2.warpAffine(img, matrix, (new_w, new_h), borderValue=(255, 255, 255))


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


def fit_to_cell(img, cell_w, cell_h):
    """Resize image to fit within cell dimensions, centering on a white background."""
    h, w = img.shape[:2]
    if w == cell_w and h == cell_h:
        return img.copy()
    canvas = np.full((cell_h, cell_w, 3), 255, dtype=np.uint8)
    x_off = (cell_w - w) // 2
    y_off = GRID_LABEL_PAD + (cell_h - GRID_LABEL_PAD - h) // 2
    canvas[y_off : y_off + h, x_off : x_off + w] = img
    return canvas


def draw_label(img, text):
    """Draw a centered label at the top of an image."""
    (tw, th), _ = cv2.getTextSize(
        text, GRID_LABEL_FONT, GRID_LABEL_SCALE, GRID_LABEL_THICKNESS
    )
    x = (img.shape[1] - tw) // 2
    y = GRID_LABEL_MARGIN + th
    cv2.putText(
        img,
        text,
        (x, y),
        GRID_LABEL_FONT,
        GRID_LABEL_SCALE,
        (0, 0, 0),
        GRID_LABEL_THICKNESS,
    )


def build_grid(panels, cols=3):
    """Arrange labeled panels into a grid. Cell size is determined by the largest panel."""
    img_h = max(p.shape[0] for _, p in panels)
    cell_w = max(p.shape[1] for _, p in panels)
    cell_h = img_h + GRID_LABEL_PAD

    cells = []
    for label, img in panels:
        cell = fit_to_cell(img, cell_w, cell_h)
        draw_label(cell, label)
        cells.append(cell)

    rows = []
    for i in range(0, len(cells), cols):
        row_cells = cells[i : i + cols]
        while len(row_cells) < cols:
            row_cells.append(np.full((cell_h, cell_w, 3), 255, dtype=np.uint8))
        rows.append(np.hstack(row_cells))
    return np.vstack(rows)


if __name__ == "__main__":
    original = load_image()
    rotated = rotate(original, ROTATION_ANGLE)
    scaled_up = scale(original, SCALE_UP_FACTOR)
    scaled_down = scale(original, SCALE_DOWN_FACTOR)

    panels = [
        ("Original", original),
        ("Scaled Up", scaled_up),
        ("Affine", original),
        ("Rotated", rotated),
        ("Scaled Down", scaled_down),
        ("Perspective", original),
    ]

    grid = build_grid(panels, cols=3)
    cv2.imshow("Lab 4: Geometric Transformations", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
