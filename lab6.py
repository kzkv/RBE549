# Tom Kazakov
# RBE 549 Lab 6: Panorama Stitching

import cv2
import numpy as np

BOSTON1_PATH = "boston1.jpeg"
BOSTON2_PATH = "boston2.jpeg"
PANORAMA_OUTPUT_PATH = "panorama.jpg"

# Lowe's ratio test
RATIO_THRESHOLD = 0.75

# Matching
KNN_NEIGHBORS = 2

# FLANN index
FLANN_INDEX_KDTREE = 1
FLANN_TREES = 5
FLANN_CHECKS = 100

# Homography
MIN_MATCH_COUNT = 10
RANSAC_REPROJ_THRESHOLD = 5.0

# Visualization
INLIER_COLOR = (0, 255, 0)
OUTLIER_COLOR = (0, 0, 255)

# cv2.Stitcher (cv2.Stitcher_PANORAMA = 0)
STITCH_MODE = 0
MAX_BUFFER_FRAMES = 10


def detect_keypoints(img):
    """Detect SIFT keypoints and compute descriptors."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(gray, None)


def match_descriptors(des1, des2):
    """Match descriptors with FLANN + Lowe's ratio test."""
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=FLANN_TREES)
    search_params = dict(checks=FLANN_CHECKS)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    raw = flann.knnMatch(des1, des2, k=KNN_NEIGHBORS)
    return [m for m, n in raw if m.distance < RATIO_THRESHOLD * n.distance]


def compute_homography(kp1, kp2, matches):
    """Estimate homography from img2 to img1 via RANSAC."""
    src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    return cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)


def draw_matches(img1, kp1, img2, kp2, matches, mask):
    """Draw side-by-side match visualization with green inliers and red outliers."""
    h1, w1 = img1.shape[:2]
    canvas = np.hstack([img1, img2])
    mask_flat = mask.ravel()
    for i, m in enumerate(matches):
        pt1 = tuple(np.int32(kp1[m.queryIdx].pt))
        pt2_raw = np.int32(kp2[m.trainIdx].pt)
        pt2 = (pt2_raw[0] + w1, pt2_raw[1])
        color = INLIER_COLOR if mask_flat[i] else OUTLIER_COLOR
        cv2.line(canvas, pt1, pt2, color, 1)
    inlier_count = int(mask.sum())
    outlier_count = len(matches) - inlier_count
    cv2.putText(
        canvas,
        f"Inliers: {inlier_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        INLIER_COLOR,
        2,
    )
    cv2.putText(
        canvas,
        f"Outliers: {outlier_count}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        OUTLIER_COLOR,
        2,
    )
    return canvas


def stitch_pair(img1, img2, H):
    """Warp img2 into img1's frame, composite, and crop to img1's vertical extent."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners2, H)

    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    all_corners = np.concatenate([corners1, warped_corners])

    x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel())

    # Translation to shift everything into positive coordinates.
    tx, ty = -x_min, -y_min
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)

    canvas_w, canvas_h = x_max - x_min, y_max - y_min
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[ty : ty + h1, tx : tx + w1] = img1
    warped = cv2.warpPerspective(img2, T @ H, (canvas_w, canvas_h))
    mask = warped.any(axis=2)
    canvas[mask] = warped[mask]

    # Crop vertically to img1's extent, then horizontally to the last full column.
    cropped = canvas[ty : ty + h1, :]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    full_columns = np.all(gray > 0, axis=0)
    right_bound = np.max(np.where(full_columns)) + 1
    return cropped[:, :right_bound]


def stitch_images(images):
    """Stitch multiple images using cv2.Stitcher."""
    stitcher = cv2.Stitcher.create(STITCH_MODE)
    status, result = stitcher.stitch(images)
    return status, result


# Camera integration
def init_state():
    """Return Lab 6 state keys with defaults."""
    return {
        "pano_buffer": [],
        "pano_result": None,
        "pano_active": False,
    }


def setup_trackbars(window_name, state):
    """No trackbars needed for Lab 6."""
    pass


def handle_key(key, state, frame):
    """Handle Lab 6 key presses. Returns True if key was handled."""
    if key == ord("n"):
        if len(state["pano_buffer"]) >= MAX_BUFFER_FRAMES:
            print(f"Buffer full ({MAX_BUFFER_FRAMES} frames)")
            return True
        state["pano_buffer"].append(frame.copy())
        print(f"Panorama buffer: {len(state['pano_buffer'])} frames")
        return True
    if key == ord("m"):
        if state["pano_active"]:
            state["pano_active"] = False
            state["pano_result"] = None
            state["pano_buffer"] = []
            print("Panorama dismissed, buffer cleared")
            return True
        if len(state["pano_buffer"]) < 2:
            print(f"Need at least 2 frames (have {len(state['pano_buffer'])})")
            return True
        status, result = stitch_images(state["pano_buffer"])
        if status == cv2.Stitcher_OK:
            state["pano_result"] = result
            state["pano_active"] = True
            print(f"Stitched {len(state['pano_buffer'])} frames")
        else:
            print(f"Stitching failed (status {status}) — try more overlap")
        return True
    return False


def apply_effects(img, state):
    """Display stitched panorama when active, otherwise pass through."""
    if not state["pano_active"] or state["pano_result"] is None:
        return img
    fh, fw = img.shape[:2]
    pano = state["pano_result"]
    ph, pw = pano.shape[:2]
    scale = min(fw / pw, fh / ph)
    resized = cv2.resize(pano, (int(pw * scale), int(ph * scale)))
    rh, rw = resized.shape[:2]
    canvas = np.zeros_like(img)
    y_off = (fh - rh) // 2
    x_off = (fw - rw) // 2
    canvas[y_off : y_off + rh, x_off : x_off + rw] = resized
    return canvas


if __name__ == "__main__":
    img1 = cv2.imread(BOSTON1_PATH)
    img2 = cv2.imread(BOSTON2_PATH)

    kp1, des1 = detect_keypoints(img1)
    kp2, des2 = detect_keypoints(img2)
    print(f"Keypoints: img1={len(kp1)}, img2={len(kp2)}")

    matches = match_descriptors(des1, des2)
    print(f"Good matches after ratio test: {len(matches)}")

    H, mask = compute_homography(kp1, kp2, matches)
    print(f"Inliers: {int(mask.sum())} / {len(matches)}")

    vis = draw_matches(img1, kp1, img2, kp2, matches, mask)
    cv2.imshow("Matches", vis)

    panorama = stitch_pair(img1, img2, H)
    cv2.imwrite(PANORAMA_OUTPUT_PATH, panorama)
    print(
        f"Panorama saved: {PANORAMA_OUTPUT_PATH} ({panorama.shape[1]}x{panorama.shape[0]})"
    )

    status, auto_panorama = stitch_images([img1, img2])
    if status == cv2.Stitcher_OK:
        print(f"Stitcher panorama: {auto_panorama.shape[1]}x{auto_panorama.shape[0]}")
    else:
        print(f"Stitcher failed with status {status}")

    cv2.imshow("Matches", vis)
    cv2.imshow("Manual Panorama", panorama)
    if status == cv2.Stitcher_OK:
        cv2.imshow("Stitcher Panorama", auto_panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
