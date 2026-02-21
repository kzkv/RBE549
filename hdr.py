import cv2
import numpy as np

# Switch between modes without changing run config: "fusion", "probe", "simulate"
RUN_MODE = "simulate"

# Part 1: input bracket and output path
BRACKET_PATHS = ["IMAGE_1.JPG", "IMAGE_2.JPG", "IMAGE_3.JPG"]
HDR_OUTPUT_PATH = "HDR.jpg"


def load_exposure_bracket(paths):
    """Load a sequence of bracketed-exposure images."""
    images = [cv2.imread(p) for p in paths]
    for path, img in zip(paths, images):
        if img is None:
            raise FileNotFoundError(f"Could not read {path}")
    shapes = {img.shape for img in images}
    if len(shapes) > 1:
        raise ValueError(f"Image dimensions do not match: {shapes}")
    return images


def align_exposures(images):
    """Align bracketed exposures in-place using median threshold bitmaps."""
    aligner = cv2.createAlignMTB()
    aligner.process(images, images)
    return images


def fuse_mertens(images):
    """Combine aligned exposures via Mertens exposure fusion."""
    merger = cv2.createMergeMertens()
    fusion = merger.process(images)
    return np.clip(fusion * 255, 0, 255).astype(np.uint8)


SETTLE_FRAMES = 15
EXPOSURE_LEVELS = [-10, -5, -2]

# Software exposure simulation: gamma > 1 darkens, gamma < 1 brightens
GAMMA_UNDER = 3.0
GAMMA_NORMAL = 1.0
GAMMA_OVER = 0.2


def probe_exposure(camera_index=0):
    """Test whether the camera supports manual exposure bracketing."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("ERROR: cannot open camera")
        return

    print(f"Backend: {cap.getBackendName()}")
    print(f"Auto exposure: {cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")
    print(f"Current exposure: {cap.get(cv2.CAP_PROP_EXPOSURE)}")

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    print(f"Auto exposure after disable: {cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")

    frames = []
    for level in EXPOSURE_LEVELS:
        cap.set(cv2.CAP_PROP_EXPOSURE, level)
        actual = cap.get(cv2.CAP_PROP_EXPOSURE)
        for _ in range(SETTLE_FRAMES):
            cap.read()
        ret, frame = cap.read()
        if not ret:
            print(f"  exposure={level}: FAILED to capture")
            continue
        mean_brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        print(
            f"  exposure={level} (actual={actual}): mean brightness={mean_brightness:.1f}"
        )
        label = f"exp={level} bright={mean_brightness:.0f}"
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frames.append(frame)

    cap.release()

    if len(frames) == 3:
        comparison = np.hstack(frames)
        cv2.imshow("Exposure Bracket Test", comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Could not capture all three exposure levels")


def build_gamma_lut(gamma):
    """Build a 256-entry lookup table for gamma correction."""
    table = np.arange(256, dtype=np.float32) / 255.0
    table = np.power(table, gamma) * 255.0
    return table.astype(np.uint8)


def adjust_exposure(image, gamma):
    """Simulate exposure change via gamma correction using a LUT."""
    return cv2.LUT(image, build_gamma_lut(gamma))


def simulate_exposure(camera_index=0):
    """Capture one frame and show software-simulated exposure bracket + fusion."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("ERROR: cannot open camera")
        return

    for _ in range(SETTLE_FRAMES):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("ERROR: failed to capture frame")
        return

    gammas = [GAMMA_UNDER, GAMMA_NORMAL, GAMMA_OVER]
    labels = ["Under", "Normal", "Over"]
    bracket = [adjust_exposure(frame, g) for g in gammas]

    fused = fuse_mertens(bracket)

    display_frames = []
    for img, label, gamma in zip(bracket, labels, gammas):
        mean_b = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        text = f"{label} (g={gamma}) bright={mean_b:.0f}"
        annotated = img.copy()
        cv2.putText(
            annotated, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        display_frames.append(annotated)

    cv2.putText(
        fused, "Mertens Fusion", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )
    display_frames.append(fused)

    top = np.hstack(display_frames[:2])
    bottom = np.hstack(display_frames[2:])
    grid = np.vstack([top, bottom])

    cv2.imshow("Simulated Exposure Bracket + Fusion", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if RUN_MODE == "probe":
        probe_exposure()
    elif RUN_MODE == "simulate":
        simulate_exposure()
    elif RUN_MODE == "fusion":
        bracket = load_exposure_bracket(BRACKET_PATHS)
        aligned = align_exposures(bracket)
        result = fuse_mertens(aligned)

        cv2.imwrite(HDR_OUTPUT_PATH, result)
        print(f"Saved {HDR_OUTPUT_PATH}")

        cv2.imshow("HDR - Mertens Fusion", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
