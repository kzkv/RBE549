# Tom Kazakov
# RBE 549 Lab 1: Experimental OpenCV App establishing the basics

import cv2
from datetime import datetime
from pathlib import Path

CAPTURES_DIR = Path("captures")
CAPTURES_DIR.mkdir(exist_ok=True)

# TODO: consider specifying camera resolution?


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    # TODO: depending on frame composition, rotation might need to be configurable or even adjustable in the app

    cv2.imshow("Lab 1 Camera", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("s"):
        now = datetime.now()
        stamped = frame.copy()
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

cap.release()
cv2.destroyAllWindows()
