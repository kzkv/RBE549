# Tom Kazakov
# RBE 549 Lab 11: Interactive point picker for crosswalk ROI

import cv2

VIDEO_PATH = "TrafficVideo.mp4"

points = []


def on_click(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    points.append((x, y))
    print(f"Point {len(points)}: ({x}, {y})")
    frame = param.copy()
    for i, pt in enumerate(points):
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        cv2.putText(frame, str(i + 1), (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    if len(points) > 1:
        cv2.polylines(frame, [__import__("numpy").array(points)], False, (0, 255, 0), 2)
    cv2.imshow("Pick Points", frame)


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

    cv2.imshow("Pick Points", frame)
    cv2.setMouseCallback("Pick Points", on_click, frame)
    print("Click crosswalk corners. Press 'q' when done.")

    while True:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print(f"\nCROSSWALK_ZONE = {points}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
