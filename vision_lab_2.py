# Tom Kazakov
# RBE 549 Lab 2: Basic and arithmetic operations on images

import cv2
import numpy as np
import time
from datetime import datetime
from pathlib import Path

CAPTURES_DIR = Path("captures")
CAPTURES_DIR.mkdir(exist_ok=True)

FLASH_DURATION_MS = 100

TIMESTAMP_FONT = cv2.FONT_HERSHEY_PLAIN
TIMESTAMP_SCALE = 2.0
TIMESTAMP_COLOR = (0, 165, 255)
TIMESTAMP_THICKNESS = 3
TIMESTAMP_PADDING = 5

BORDER_SIZE = 18
BORDER_COLOR = (0, 0, 255)

LOGO_PATH = Path("OpenCV.png")
LOGO_HEIGHT = 80
LOGO_ALPHA = 0.6

COLOR_SAMPLE_KERNEL = 11
COLOR_HUE_TOLERANCE = 10
COLOR_SAT_TOLERANCE = 50
COLOR_VAL_TOLERANCE = 50

ROTATION_STEP = 10


def load_logo():
    """Load and resize logo for blending."""
    logo = cv2.imread(str(LOGO_PATH))
    if logo is None:
        return None

    aspect = logo.shape[1] / logo.shape[0]
    return cv2.resize(logo, (int(LOGO_HEIGHT * aspect), LOGO_HEIGHT))


def sample_color_at(img, x, y, kernel_size=COLOR_SAMPLE_KERNEL):
    """Sample average HSV color from a kernel region around (x, y)."""
    h, w = img.shape[:2]
    half = kernel_size // 2
    x1, y1 = max(0, x - half), max(0, y - half)
    x2, y2 = min(w, x + half + 1), min(h, y + half + 1)

    region = img[y1:y2, x1:x2]
    hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    avg_hsv = hsv_region.mean(axis=(0, 1)).astype(int)
    return avg_hsv


def get_color_mask(img, target_hsv):
    """Create a mask for pixels matching the target HSV color within tolerances."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array(
        [
            max(0, target_hsv[0] - COLOR_HUE_TOLERANCE),
            max(0, target_hsv[1] - COLOR_SAT_TOLERANCE),
            max(0, target_hsv[2] - COLOR_VAL_TOLERANCE),
        ]
    )
    upper = np.array(
        [
            min(179, target_hsv[0] + COLOR_HUE_TOLERANCE),
            min(255, target_hsv[1] + COLOR_SAT_TOLERANCE),
            min(255, target_hsv[2] + COLOR_VAL_TOLERANCE),
        ]
    )
    return cv2.inRange(hsv, lower, upper)


def draw_timestamp(img, include_seconds=True):
    """Draw timestamp at bottom right and return its ROI bounds (y1, y2, x1, x2)."""
    h, w = img.shape[:2]
    fmt = "%m %d '%y  %H:%M:%S" if include_seconds else "%m %d '%y  %H:%M"
    text = datetime.now().strftime(fmt)

    (text_w, text_h), baseline = cv2.getTextSize(
        text, TIMESTAMP_FONT, TIMESTAMP_SCALE, TIMESTAMP_THICKNESS
    )

    x = w - text_w - TIMESTAMP_PADDING
    y = h - TIMESTAMP_PADDING - baseline
    cv2.putText(
        img,
        text,
        (x, y),
        TIMESTAMP_FONT,
        TIMESTAMP_SCALE,
        TIMESTAMP_COLOR,
        TIMESTAMP_THICKNESS,
    )

    roi_x1 = x - TIMESTAMP_PADDING
    roi_x2 = w
    roi_y1 = y - text_h - TIMESTAMP_PADDING
    roi_y2 = h
    return roi_y1, roi_y2, roi_x1, roi_x2


def apply_zoom(img, zoom_pct):
    if zoom_pct == 100:
        return img
    h, w = img.shape[:2]
    crop_h, crop_w = int(h * 100 / zoom_pct), int(w * 100 / zoom_pct)
    y1, x1 = (h - crop_h) // 2, (w - crop_w) // 2
    cropped = img[y1 : y1 + crop_h, x1 : x1 + crop_w]
    return cv2.resize(cropped, (w, h))


cap = cv2.VideoCapture(0)
fps = 30

THRESHOLD_TYPES = [
    (cv2.THRESH_BINARY, "BINARY"),
    (cv2.THRESH_BINARY_INV, "BINARY_INV"),
    (cv2.THRESH_TRUNC, "TRUNC"),
    (cv2.THRESH_TOZERO, "TOZERO"),
    (cv2.THRESH_TOZERO_INV, "TOZERO_INV"),
]

state = {
    "zoom": 0,
    "rotation": 0,
    "extract_mode": None,  # None, "sampling", or "extracting"
    "mouse_pos": (0, 0),
    "target_hsv": None,
    "threshold_idx": None,  # None = off, 0-4 = index into THRESHOLD_TYPES
}


def on_mouse(event, x, y, flags, param):
    state["mouse_pos"] = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN and state["extract_mode"] == "sampling":
        state["extract_mode"] = "extracting"


cv2.namedWindow("Lab 2 Camera")
cv2.setMouseCallback("Lab 2 Camera", on_mouse)
cv2.createTrackbar("Zoom, %: ", "Lab 2 Camera", 0, 100, lambda v: state.update(zoom=v))

video = None
flash_start_time = None

logo = load_logo()
if logo is not None:
    logo_h, logo_w = logo.shape[:2]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    zoom_pct = 100 + state["zoom"]
    zoomed = apply_zoom(frame, zoom_pct)
    display = zoomed.copy()

    # Apply rotation
    if state["rotation"] != 0:
        h, w = display.shape[:2]
        center = ((w - 1) / 2.0, (h - 1) / 2.0)
        rot_matrix = cv2.getRotationMatrix2D(center, state["rotation"], 1.0)
        display = cv2.warpAffine(display, rot_matrix, (w, h))

    # Color extraction mode
    if state["extract_mode"] == "sampling":
        mx, my = state["mouse_pos"]
        mx = max(0, min(mx - BORDER_SIZE, display.shape[1] - 1))
        my = max(0, min(my - BORDER_SIZE, display.shape[0] - 1))
        state["target_hsv"] = sample_color_at(display, mx, my)
    elif state["extract_mode"] == "extracting" and state["target_hsv"] is not None:
        mask = get_color_mask(display, state["target_hsv"])
        display = cv2.bitwise_and(display, display, mask=mask)

    # Apply thresholding
    if state["threshold_idx"] is not None:
        thresh_type, _ = THRESHOLD_TYPES[state["threshold_idx"]]
        gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
        _, threshed = cv2.threshold(gray, 127, 255, thresh_type)
        display = cv2.cvtColor(threshed, cv2.COLOR_GRAY2BGR)

    if video:
        if int(time.time()) % 3:
            cv2.circle(display, (display.shape[1] - 20, 20), 8, (0, 0, 255), -1)
        stamped = zoomed.copy()
        draw_timestamp(stamped, include_seconds=True)
        video.write(stamped)

    # Draw timestamp and copy its ROI to top right
    roi_y1, roi_y2, roi_x1, roi_x2 = draw_timestamp(display, include_seconds=True)
    timestamp_roi = display[roi_y1:roi_y2, roi_x1:roi_x2].copy()
    roi_h, roi_w = timestamp_roi.shape[:2]
    display[0:roi_h, display.shape[1] - roi_w : display.shape[1]] = timestamp_roi

    # Blend logo at top left using addWeighted
    if logo is not None:
        roi = display[0:logo_h, 0:logo_w]
        blended = cv2.addWeighted(roi, 1 - LOGO_ALPHA, logo, LOGO_ALPHA, 0)
        display[0:logo_h, 0:logo_w] = blended

    # Apply flash effect if within flash duration
    if flash_start_time is not None:
        elapsed_ms = (time.time() - flash_start_time) * 1000
        if elapsed_ms < FLASH_DURATION_MS:
            display[:, :] = (255, 255, 255)
        else:
            flash_start_time = None

    help_text = "Esc: quit  c: capture  v: record  +/-: zoom  e: extract  r/R: rotate  t/T: threshold"
    (help_w, help_h), help_baseline = cv2.getTextSize(
        help_text, cv2.FONT_HERSHEY_PLAIN, 1.0, 1
    )
    help_pad = 5
    disp_h = display.shape[0]
    overlay = display.copy()
    cv2.rectangle(
        overlay,
        (help_pad, disp_h - help_h - 2 * help_pad),
        (help_w + 2 * help_pad, disp_h - help_pad),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)
    cv2.putText(
        display,
        help_text,
        (help_pad + 5, disp_h - help_pad - help_baseline),
        cv2.FONT_HERSHEY_PLAIN,
        1.0,
        (230, 230, 230),
        1,
    )

    # Show sampling preview with kernel outline and color swatch
    if state["extract_mode"] == "sampling" and state["target_hsv"] is not None:
        mx, my = state["mouse_pos"]
        mx = max(0, min(mx - BORDER_SIZE, display.shape[1] - 1))
        my = max(0, min(my - BORDER_SIZE, display.shape[0] - 1))
        half = COLOR_SAMPLE_KERNEL // 2
        cv2.rectangle(
            display, (mx - half, my - half), (mx + half, my + half), (0, 255, 0), 2
        )
        bgr_color = cv2.cvtColor(
            np.array([[state["target_hsv"]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
        )[0, 0]
        cv2.rectangle(
            display,
            (mx + half + 5, my - half),
            (mx + half + 35, my + half),
            bgr_color.tolist(),
            -1,
        )
        cv2.rectangle(
            display,
            (mx + half + 5, my - half),
            (mx + half + 35, my + half),
            (255, 255, 255),
            1,
        )

    display = cv2.copyMakeBorder(
        display,
        BORDER_SIZE,
        BORDER_SIZE,
        BORDER_SIZE,
        BORDER_SIZE,
        cv2.BORDER_CONSTANT,
        value=BORDER_COLOR,
    )

    cv2.imshow("Lab 2 Camera", display)

    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key in (ord("+"), ord("=")):
        state["zoom"] = min(100, state["zoom"] + 10)
        cv2.setTrackbarPos("Zoom, %: ", "Lab 2 Camera", state["zoom"])
    elif key in (ord("-"), ord("_")):
        state["zoom"] = max(0, state["zoom"] - 10)
        cv2.setTrackbarPos("Zoom, %: ", "Lab 2 Camera", state["zoom"])
    elif key == ord("c"):
        flash_start_time = time.time()
        stamped = zoomed.copy()
        if state["rotation"] != 0:
            h, w = stamped.shape[:2]
            center = ((w - 1) / 2.0, (h - 1) / 2.0)
            rot_matrix = cv2.getRotationMatrix2D(center, state["rotation"], 1.0)
            stamped = cv2.warpAffine(stamped, rot_matrix, (w, h))
        if state["extract_mode"] == "extracting" and state["target_hsv"] is not None:
            mask = get_color_mask(stamped, state["target_hsv"])
            stamped = cv2.bitwise_and(stamped, stamped, mask=mask)
        if state["threshold_idx"] is not None:
            thresh_type, _ = THRESHOLD_TYPES[state["threshold_idx"]]
            gray = cv2.cvtColor(stamped, cv2.COLOR_BGR2GRAY)
            _, threshed = cv2.threshold(gray, 127, 255, thresh_type)
            stamped = cv2.cvtColor(threshed, cv2.COLOR_GRAY2BGR)
        draw_timestamp(stamped, include_seconds=False)
        filename = CAPTURES_DIR / datetime.now().strftime("lab2_%Y-%m-%d_%H-%M-%S.jpg")
        cv2.imwrite(str(filename), stamped)
        print(f"Saved: {filename}")
    elif key == ord("e"):
        if state["extract_mode"] is None:
            state["extract_mode"] = "sampling"
            print("Color extraction: sampling mode - click to select color")
        else:
            state["extract_mode"] = None
            state["target_hsv"] = None
            print("Color extraction: disabled")
    elif key == ord("r"):
        state["rotation"] = (state["rotation"] - ROTATION_STEP) % 360
        print(f"Rotation: {state['rotation']}° (CW)")
    elif key == ord("R"):
        state["rotation"] = (state["rotation"] + ROTATION_STEP) % 360
        print(f"Rotation: {state['rotation']}° (CCW)")
    elif key == ord("t"):
        if state["threshold_idx"] is None:
            state["threshold_idx"] = 0
        else:
            state["threshold_idx"] = (state["threshold_idx"] + 1) % len(THRESHOLD_TYPES)
        _, name = THRESHOLD_TYPES[state["threshold_idx"]]
        print(f"Threshold: {name}")
    elif key == ord("T"):
        state["threshold_idx"] = None
        print("Threshold: disabled")
    elif key == ord("v"):
        if video:
            video.release()
            video = None
            print("Recording stopped")
        else:
            h, w = zoomed.shape[:2]
            filename = CAPTURES_DIR / datetime.now().strftime(
                "lab2_%Y-%m-%d_%H-%M-%S.mp4"
            )
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video = cv2.VideoWriter(str(filename), fourcc, fps, (w, h))
            print(f"Recording: {filename} @ {fps}fps")

if video:
    video.release()
cap.release()
cv2.destroyAllWindows()
