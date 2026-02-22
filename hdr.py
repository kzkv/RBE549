import threading

import cv2
import numpy as np

# Part 1: input bracket and output path
BRACKET_PATHS = ["IMAGE_1.JPG", "IMAGE_2.JPG", "IMAGE_3.JPG"]
HDR_OUTPUT_PATH = "HDR.jpg"

# Software exposure simulation: gamma > 1 darkens, gamma < 1 brightens
GAMMA_UNDER = 3.0
GAMMA_NORMAL = 1.0
GAMMA_OVER = 0.2


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


def build_gamma_lut(gamma):
    """Build a 256-entry lookup table for gamma correction."""
    table = np.arange(256, dtype=np.float32) / 255.0
    table = np.power(table, gamma) * 255.0
    return table.astype(np.uint8)


class CaptureWorker:
    """One thread in the bracket capture system, responsible for one exposure level."""

    def __init__(self, gamma, cap, cap_lock, barrier, stop_event):
        self.lut = build_gamma_lut(gamma)
        self.cap = cap
        self.cap_lock = cap_lock
        self.barrier = barrier
        self.stop_event = stop_event
        self.frame = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            with self.cap_lock:
                ret, raw = self.cap.read()
            if not ret:
                continue
            self.frame = cv2.LUT(raw, self.lut)
            try:
                self.barrier.wait(timeout=1.0)
            except threading.BrokenBarrierError:
                break


class BracketCapture:
    """Synchronized three-thread exposure bracket capture from a single camera."""

    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.barrier = threading.Barrier(4)
        gammas = [GAMMA_UNDER, GAMMA_NORMAL, GAMMA_OVER]
        self.workers = [
            CaptureWorker(g, self.cap, self.cap_lock, self.barrier, self.stop_event)
            for g in gammas
        ]

    def start(self):
        for w in self.workers:
            w.start()

    def get_triplet(self):
        """Block until all three workers have a frame, then return (under, normal, over)."""
        try:
            self.barrier.wait(timeout=1.0)
        except threading.BrokenBarrierError:
            return None
        return tuple(w.frame for w in self.workers)

    def stop(self):
        self.stop_event.set()
        self.barrier.abort()
        for w in self.workers:
            w.thread.join(timeout=2.0)
        self.cap.release()


def run_hdr_capture(camera_index=0):
    """Live preview of three synchronized exposure streams."""
    bracket_cap = BracketCapture(camera_index)
    if not bracket_cap.cap.isOpened():
        print("ERROR: cannot open camera")
        return

    bracket_cap.start()
    print("HDR capture running. Press 'q' to quit.")

    labels = ["Under", "Normal", "Over"]
    while True:
        triplet = bracket_cap.get_triplet()
        if triplet is None:
            continue

        panels = []
        for frame, label in zip(triplet, labels):
            annotated = frame.copy()
            cv2.putText(
                annotated,
                label,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            panels.append(annotated)

        display = np.hstack(panels)
        cv2.imshow("HDR Bracket Capture", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    bracket_cap.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_hdr_capture()
