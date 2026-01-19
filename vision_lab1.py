# Tom Kazakov
# RBE 549 Lab 1: Experimental OpenCV App establishing the basics

import cv2
import time
from datetime import datetime
from pathlib import Path

CAPTURES_DIR = Path("captures")
CAPTURES_DIR.mkdir(exist_ok=True)


def apply_zoom(img, zoom_pct):
    if zoom_pct == 100:
        return img
    h, w = img.shape[:2]
    crop_h, crop_w = int(h * 100 / zoom_pct), int(w * 100 / zoom_pct)
    y1, x1 = (h - crop_h) // 2, (w - crop_w) // 2
    cropped = img[y1 : y1 + crop_h, x1 : x1 + crop_w]
    return cv2.resize(cropped, (w, h))


cap = cv2.VideoCapture(0)
zoom_pct = 100
video = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    zoomed = apply_zoom(frame, zoom_pct)
    display = zoomed.copy()

    if video:
        if int(time.time()) % 3:  # Blinking recording indicator light
            cv2.circle(display, (display.shape[1] - 20, 20), 8, (0, 0, 255), -1)
        video.write(zoomed)

    help_lines = ["Esc: quit  s: save  v: record", f"+/-: zoom ({zoom_pct}%)"]
    overlay = display.copy()
    cv2.rectangle(overlay, (5, 5), (170, 45), (0, 0, 0), -1)
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

    cv2.imshow("Lab 1 Camera", display)

    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key in (ord("+"), ord("=")):
        zoom_pct = min(200, zoom_pct + 10)
    elif key in (ord("-"), ord("_")):
        zoom_pct = max(100, zoom_pct - 10)
    elif key == ord("s"):
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
        filename = CAPTURES_DIR / now.strftime("lab1_%Y-%m-%d_%H-%M-%S.jpg")
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
                "lab1_%Y-%m-%d_%H-%M-%S.mp4"
            )
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video = cv2.VideoWriter(str(filename), fourcc, 30, (w, h))
            print(f"Recording: {filename}")

if video:
    video.release()
cap.release()
cv2.destroyAllWindows()
