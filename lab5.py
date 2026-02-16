# Tom Kazakov
# RBE 549 Lab 5: Feature Matching & Object Detection
#
# SURF requires OpenCV built with NONFREE flag:
# CMAKE_ARGS="-DOPENCV_ENABLE_NONFREE=ON" pip install --no-binary=opencv-contrib-python opencv-contrib-python

import cv2

BOOK_PATH = "book.jpg"
TABLE_PATH = "table.jpg"
RATIO_THRESHOLD = 0.5


def detect_sift(img):
    """Detect SIFT keypoints and compute descriptors."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(gray, None)


def match_bruteforce(des1, des2):
    """Match descriptors with BFMatcher + Lowe's ratio test."""
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(des1, des2, k=2)
    return [m for m, n in raw if m.distance < RATIO_THRESHOLD * n.distance]


if __name__ == "__main__":
    book = cv2.imread(BOOK_PATH)
    table = cv2.imread(TABLE_PATH)

    kp1, des1 = detect_sift(book)
    kp2, des2 = detect_sift(table)
    good = match_bruteforce(des1, des2)

    result = cv2.drawMatches(
        book,
        kp1,
        table,
        kp2,
        good,
        None,
        matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    label = f"SIFT + BF | ratio {RATIO_THRESHOLD} | {len(good)} matches"
    cv2.putText(result, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    print(label)
    cv2.imshow("SIFT + Brute-Force", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
