import cv2
import numpy as np

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


if __name__ == "__main__":
    bracket = load_exposure_bracket(BRACKET_PATHS)
    aligned = align_exposures(bracket)
    result = fuse_mertens(aligned)

    cv2.imwrite(HDR_OUTPUT_PATH, result)
    print(f"Saved {HDR_OUTPUT_PATH}")

    cv2.imshow("HDR - Mertens Fusion", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
