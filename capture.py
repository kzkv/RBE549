# Capture script for testing resolution and FOV

import cv2
from datetime import datetime
from pathlib import Path

CAPTURES_DIR = Path("captures")
CAPTURES_DIR.mkdir(exist_ok=True)

PREVIEW_SIZE = (1280, 720)
ZOOM_CROP = (1920, 1080)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Streaming at {actual_w}x{actual_h}")
print("Press 'z' to toggle zoom mode (1:1 center crop)")
print("Press 'c' to capture, Esc to quit")

zoom_mode = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if zoom_mode:
        cx, cy = actual_w // 2, actual_h // 2
        x1 = cx - ZOOM_CROP[0] // 2
        y1 = cy - ZOOM_CROP[1] // 2
        display = frame[y1 : y1 + ZOOM_CROP[1], x1 : x1 + ZOOM_CROP[0]]
        window_name = "Zoom Mode (1:1 center crop)"
    else:
        display = cv2.resize(frame, PREVIEW_SIZE)
        window_name = "Preview (scaled)"

    cv2.imshow(window_name, display)

    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord("z"):
        zoom_mode = not zoom_mode
        cv2.destroyAllWindows()
        print(f"Zoom mode: {'ON' if zoom_mode else 'OFF'}")
    elif key == ord("c"):
        filename = CAPTURES_DIR / datetime.now().strftime(
            f"capture_{actual_w}x{actual_h}_%H%M%S.jpg"
        )
        cv2.imwrite(str(filename), frame)
        print(f"Saved: {filename}")

cap.release()
cv2.destroyAllWindows()
