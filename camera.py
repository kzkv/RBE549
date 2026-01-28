# Tom Kazakov
# RBE 549 Camera App - Main wrapper for Labs 1 and 2

import cv2
from datetime import datetime
from pathlib import Path

import lab1
import lab2

WINDOW_NAME = "RBE 549 Camera"
CAPTURES_DIR = Path("captures")
CAPTURES_DIR.mkdir(exist_ok=True)

FPS = 30  # BRIO reports incorrect FPS (5.0); actual rate is 30


def create_state():
    """Initialize combined state from all labs."""
    state = {}
    state.update(lab1.init_state())
    state.update(lab2.init_state())
    return state


def draw_help_text(display):
    """Draw keyboard shortcuts legend at bottom left."""
    help_text = "Esc: quit  c: capture  v: record  +/-: zoom  e: extract  r/R: rotate  t/T: thresh  b: blur  s: sharp"
    (help_w, help_h), help_baseline = cv2.getTextSize(
        help_text, cv2.FONT_HERSHEY_PLAIN, 1.0, 1
    )
    help_pad = 5
    disp_h = display.shape[0]
    overlay = display.copy()
    cv2.rectangle(
        overlay,
        (help_pad, disp_h - help_h - 2 * help_pad),
        (help_w + 2 * help_pad, disp_h - help_pad),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)
    cv2.putText(
        display,
        help_text,
        (help_pad + 5, disp_h - help_pad - help_baseline),
        cv2.FONT_HERSHEY_PLAIN,
        1.0,
        (230, 230, 230),
        1,
    )


def main():
    cap = cv2.VideoCapture(1)
    state = create_state()

    cv2.namedWindow(WINDOW_NAME)
    lab1.setup_trackbars(WINDOW_NAME, state)
    lab2.setup_trackbars(WINDOW_NAME, state)
    lab2.setup_mouse_callback(WINDOW_NAME, state)

    logo, logo_h, logo_w = lab2.load_logo()
    video = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        zoom_pct = 100 + state["zoom"]
        zoomed = lab1.apply_zoom(frame, zoom_pct)

        lab2.update_color_sample(zoomed, state)

        display = lab2.apply_effects(zoomed, state)

        # Video recording
        if video:
            stamped = lab2.apply_effects(zoomed, state)
            lab1.draw_timestamp(stamped, include_seconds=True)
            video.write(stamped)

        # Draw timestamp and copy ROI to top right
        roi_bounds = lab1.draw_timestamp(display, include_seconds=True)
        roi_h = lab2.copy_timestamp_roi(display, roi_bounds)

        # Draw overlays
        lab2.draw_overlays(
            display, state, logo, logo_h, logo_w, roi_h, video is not None
        )

        # Flash effect
        lab2.apply_flash(display, state)

        # Help text
        draw_help_text(display)

        # Border
        display = lab2.apply_border(display)

        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1)
        if key == 27:
            break

        # Lab 1 keys
        if lab1.handle_key(key, state, WINDOW_NAME):
            continue

        # Lab 2 keys
        if lab2.handle_key(key, state):
            continue

        # Shared keys
        if key == ord("c"):
            lab2.trigger_flash(state)
            stamped = lab2.apply_effects(zoomed, state)
            lab1.draw_timestamp(stamped, include_seconds=False)
            filename = CAPTURES_DIR / datetime.now().strftime(
                "capture_%Y-%m-%d_%H-%M-%S.jpg"
            )
            cv2.imwrite(str(filename), stamped)
            print(f"Saved: {filename}")

        elif key == ord("v"):
            if video:
                video.release()
                video = None
                print("Recording stopped")
            else:
                h, w = zoomed.shape[:2]
                filename = CAPTURES_DIR / datetime.now().strftime(
                    "video_%Y-%m-%d_%H-%M-%S.mp4"
                )
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video = cv2.VideoWriter(str(filename), fourcc, FPS, (w, h))
                print(f"Recording: {filename} @ {FPS}fps")

    if video:
        video.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
