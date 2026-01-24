# Tom Kazakov
# RBE 549 Lab 2: Basic and arithmetic operations on images

import cv2
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

state = {"zoom": 0}
cv2.namedWindow("Lab 2 Camera")
cv2.createTrackbar("Zoom, %: ", "Lab 2 Camera", 0, 100, lambda v: state.update(zoom=v))

video = None
flash_start_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    zoom_pct = 100 + state["zoom"]
    zoomed = apply_zoom(frame, zoom_pct)
    display = zoomed.copy()

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

    # Apply flash effect if within flash duration
    if flash_start_time is not None:
        elapsed_ms = (time.time() - flash_start_time) * 1000
        if elapsed_ms < FLASH_DURATION_MS:
            display[:, :] = (255, 255, 255)
        else:
            flash_start_time = None

    help_text = "Esc: quit  c: capture  v: record  +/-: zoom"
    (help_w, help_h), help_baseline = cv2.getTextSize(
        help_text, cv2.FONT_HERSHEY_PLAIN, 1.0, 1
    )
    help_pad = 5
    overlay = display.copy()
    cv2.rectangle(
        overlay,
        (help_pad, help_pad),
        (help_w + 2 * help_pad, help_h + 2 * help_pad),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)
    cv2.putText(
        display,
        help_text,
        (help_pad + 5, help_h + help_pad),
        cv2.FONT_HERSHEY_PLAIN,
        1.0,
        (230, 230, 230),
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
        draw_timestamp(stamped, include_seconds=False)
        filename = CAPTURES_DIR / datetime.now().strftime("lab2_%Y-%m-%d_%H-%M-%S.jpg")
        cv2.imwrite(str(filename), stamped)
        print(f"Saved: {filename}")
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
