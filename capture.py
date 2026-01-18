# Capture script for testing resolution and FOV

import cv2
from datetime import datetime
from pathlib import Path

CAPTURES_DIR = Path("captures")
CAPTURES_DIR.mkdir(exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

for _ in range(10):
    ret, frame = cap.read()
cap.release()

if ret:
    h, w = frame.shape[:2]
    filename = CAPTURES_DIR / datetime.now().strftime(f"capture_{w}x{h}_%H%M%S.jpg")
    cv2.imwrite(str(filename), frame)
    print(f"Saved: {filename}")
else:
    print("Failed to capture")
