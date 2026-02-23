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

THUMBNAIL_SCALE = 0.15
HELP_BAR_HEIGHT = 45
WATERMARK_TEXT = "HDR"


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


CONTRAST_WEIGHT = 1.0
SATURATION_WEIGHT = 1.0
EXPOSURE_WEIGHT = 1.0


def fuse_mertens(images, cw=CONTRAST_WEIGHT, sw=SATURATION_WEIGHT, ew=EXPOSURE_WEIGHT):
    """Combine aligned exposures via Mertens exposure fusion."""
    merger = cv2.createMergeMertens(cw, sw, ew)
    fusion = merger.process(images)
    return np.clip(fusion * 255, 0, 255).astype(np.uint8)


def build_gamma_lut(gamma):
    """Build a 256-entry lookup table for gamma correction."""
    table = np.arange(256, dtype=np.float32) / 255.0
    table = np.power(table, gamma) * 255.0
    return table.astype(np.uint8)


class ExposureWorker:
    """One thread in the bracket system, applies a fixed gamma LUT."""

    def __init__(self, gamma, bracket):
        self.lut = build_gamma_lut(gamma)
        self.bracket = bracket
        self.frame = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        while not self.bracket.stop_event.is_set():
            try:
                self.bracket.start_barrier.wait(timeout=1.0)
            except threading.BrokenBarrierError:
                break
            if self.bracket.stop_event.is_set():
                break
            self.frame = cv2.LUT(self.bracket.source_frame, self.lut)
            try:
                self.bracket.done_barrier.wait(timeout=1.0)
            except threading.BrokenBarrierError:
                break


class ExposureBracket:
    """Three-thread exposure bracket processor using two-barrier synchronization."""

    def __init__(self):
        self.stop_event = threading.Event()
        self.start_barrier = threading.Barrier(4)
        self.done_barrier = threading.Barrier(4)
        self.source_frame = None
        gammas = [GAMMA_UNDER, GAMMA_NORMAL, GAMMA_OVER]
        self.workers = [ExposureWorker(g, self) for g in gammas]

    def start(self):
        for w in self.workers:
            w.start()

    def process(self, frame):
        """Post a frame, wait for all three workers, return the triplet."""
        self.source_frame = frame
        try:
            self.start_barrier.wait(timeout=1.0)
            self.done_barrier.wait(timeout=1.0)
        except threading.BrokenBarrierError:
            return None
        return [w.frame for w in self.workers]

    def stop(self):
        self.stop_event.set()
        self.start_barrier.abort()
        self.done_barrier.abort()
        for w in self.workers:
            w.thread.join(timeout=2.0)


def add_watermark(image):
    """Draw semi-transparent HDR watermark in the top-right corner."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.5
    thickness = 3
    (tw, th), _ = cv2.getTextSize(WATERMARK_TEXT, font, scale, thickness)
    h, w = image.shape[:2]
    x = w - tw - 20
    y = th + 20
    overlay = image.copy()
    cv2.putText(overlay, WATERMARK_TEXT, (x, y), font, scale, (0, 255, 0), thickness)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    return image


def draw_thumbnails(display, triplet):
    """Draw under/normal/over thumbnails in the bottom-left, above the help bar."""
    h, w = display.shape[:2]
    labels = ["Under", "Normal", "Over"]
    thumb_w = int(w * THUMBNAIL_SCALE)

    thumbs = []
    for frame, label in zip(triplet, labels):
        fh, fw = frame.shape[:2]
        thumb_h = int(fh * thumb_w / fw)
        thumb = cv2.resize(frame, (thumb_w, thumb_h))
        cv2.putText(thumb, label, (4, 14), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 1)
        thumbs.append(thumb)

    strip = np.hstack(thumbs)
    sh, sw = strip.shape[:2]

    pad = 5
    y_start = h - HELP_BAR_HEIGHT - sh - pad
    x_start = pad

    if y_start >= 0 and x_start + sw <= w:
        display[y_start : y_start + sh, x_start : x_start + sw] = strip


def init_state():
    return {
        "hdr_active": False,
        "hdr_bracket": None,
    }


def setup_trackbars(window_name, state):
    pass


def handle_key(key, state):
    if key != ord("h"):
        return False
    state["hdr_active"] = not state["hdr_active"]
    if state["hdr_active"]:
        if state["hdr_bracket"] is None:
            state["hdr_bracket"] = ExposureBracket()
            state["hdr_bracket"].start()
        print("HDR mode ON")
    else:
        if state["hdr_bracket"] is not None:
            state["hdr_bracket"].stop()
            state["hdr_bracket"] = None
        print("HDR mode OFF")
    return True


def apply_effects(img, state):
    """When HDR is active, fuse the bracket and overlay thumbnails."""
    if not state["hdr_active"] or state["hdr_bracket"] is None:
        return img
    triplet = state["hdr_bracket"].process(img)
    if triplet is None:
        return img
    fused = fuse_mertens(triplet)
    fused = add_watermark(fused)
    draw_thumbnails(fused, triplet)
    return fused


WEIGHT_SWEEP = [
    (1.0, 1.0, 0.0),
    (1.0, 1.0, 0.5),
    (1.0, 1.0, 1.0),
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (1.0, 1.0, 2.0),
]


CROP_Y = 0
CROP_H = 600
CROP_X = 400
CROP_W = 800


def run_weight_sweep(images):
    """Fuse with several weight combinations and display a 1:1 crop comparison grid."""
    cols = 3
    rows = (len(WEIGHT_SWEEP) + cols - 1) // cols

    grid = np.zeros((rows * CROP_H, cols * CROP_W, 3), dtype=np.uint8)

    for idx, (cw, sw, ew) in enumerate(WEIGHT_SWEEP):
        fused = fuse_mertens(images, cw, sw, ew)
        crop = fused[CROP_Y : CROP_Y + CROP_H, CROP_X : CROP_X + CROP_W]
        label = f"cw={cw} sw={sw} ew={ew}"
        cv2.putText(crop, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        r, c = divmod(idx, cols)
        grid[r * CROP_H : (r + 1) * CROP_H, c * CROP_W : (c + 1) * CROP_W] = crop

    return grid


if __name__ == "__main__":
    bracket = load_exposure_bracket(BRACKET_PATHS)
    aligned = align_exposures(bracket)

    result = fuse_mertens(aligned)
    cv2.imwrite(HDR_OUTPUT_PATH, result)
    print(f"Saved {HDR_OUTPUT_PATH}")

    grid = run_weight_sweep(aligned)
    cv2.imshow("Mertens Weight Sweep", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
