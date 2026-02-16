# Tom Kazakov
# RBE 549 Lab 5: Feature Matching & Object Detection
#
# SURF requires OpenCV built with NONFREE flag:
# CMAKE_ARGS="-DOPENCV_ENABLE_NONFREE=ON" pip install --no-binary=opencv-contrib-python opencv-contrib-python

import cv2


def detect_sift(img):
    """Detect SIFT keypoints and compute descriptors."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(gray, None)
