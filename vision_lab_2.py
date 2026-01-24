# Tom Kazakov
# RBE 549 Lab 2: Basic and arithmetic operations on images

import cv2
import time
from datetime import datetime
from pathlib import Path

CAPTURES_DIR = Path("captures")
CAPTURES_DIR.mkdir(exist_ok=True)

FLASH_DURATION_MS = 100


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
        h, w = zoomed.shape[:2]
        stamped = zoomed.copy()
        timestamp = datetime.now().strftime("%m %d '%y  %H:%M:%S")
        cv2.putText(
            stamped,
            timestamp,
            (w - 380, h - 30),
            cv2.FONT_HERSHEY_PLAIN,
            2.0,
            (0, 165, 255),
            3,
        )
        video.write(stamped)

    # Apply flash effect if within flash duration
    if flash_start_time is not None:
        elapsed_ms = (time.time() - flash_start_time) * 1000
        if elapsed_ms < FLASH_DURATION_MS:
            display[:, :] = (255, 255, 255)
        else:
            flash_start_time = None

    help_lines = ["Esc: quit  c: capture  v: record  +/-: zoom"]
    overlay = display.copy()
    cv2.rectangle(overlay, (5, 5), (395, 25), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)
    for i, line in enumerate(help_lines):
        cv2.putText(
            display,
            line,
            (10, 20 + i * 18),
            cv2.FONT_HERSHEY_PLAIN,
            1.0,
            (230, 230, 230),
            1,
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
        now = datetime.now()
        stamped = zoomed.copy()
        h, w = stamped.shape[:2]
        timestamp = now.strftime("%m %d '%y  %H:%M")
        cv2.putText(
            stamped,
            timestamp,
            (w - 350, h - 30),
            cv2.FONT_HERSHEY_PLAIN,
            2.0,
            (0, 165, 255),
            3,
        )
        filename = CAPTURES_DIR / now.strftime("lab2_%Y-%m-%d_%H-%M-%S.jpg")
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
