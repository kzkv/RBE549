# Tom Kazakov
# RBE 549 Lab 12: Real-time Neural Style Transfer via TF Hub

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

CAMERA_INDEX = 0
STYLE_IMAGES = {"1": "kandinsky1.jpg", "2": "kandinsky2.jpg"}
CONTENT_SIZE = 384
HUB_MODEL_URL = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"

VERIFY_CONTENT_PATH = "YellowLabradorLooking_new.jpg"


def load_and_preprocess(path, target_size=None):
    """Load an image, convert to float32 [0, 1], and add batch dimension."""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, dtype=tf.float32)
    if target_size is not None:
        img = tf.image.resize(img, target_size)
    return img[tf.newaxis, ...]


def load_hub_model():
    """Load the Magenta arbitrary style transfer model from TF Hub."""
    return hub.load(HUB_MODEL_URL)


def stylize_frame(model, frame, style_tensor):
    """Stylize a BGR OpenCV frame and return the result as BGR uint8."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = CONTENT_SIZE / max(h, w)
    small = cv2.resize(rgb, (int(w * scale), int(h * scale)))
    content_tensor = tf.constant(small[np.newaxis, ...].astype("float32") / 255.0)
    output = model(content_tensor, style_tensor)[0]
    result = tf.squeeze(output).numpy()
    result = np.clip(result * 255, 0, 255).astype("uint8")
    result = cv2.resize(result, (w, h))
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def verify_with_tutorial_images(model):
    """Stylize the labrador with Kandinsky and display as a sanity check."""
    content = load_and_preprocess(VERIFY_CONTENT_PATH)
    style = load_and_preprocess(STYLE_IMAGES["1"], target_size=(256, 256))
    output = model(content, style)[0]
    result = tf.squeeze(output).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(tf.squeeze(content).numpy())
    axes[0].set_title("Content")
    axes[0].axis("off")
    axes[1].imshow(tf.squeeze(style).numpy())
    axes[1].set_title("Style")
    axes[1].axis("off")
    axes[2].imshow(result)
    axes[2].set_title("Stylized")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig("nst_verification.png")
    plt.show()


def run_camera(model, style_tensors):
    """Run the camera loop with real-time style transfer."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    stylize_on = True
    active_style = "1"
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if stylize_on:
            display = stylize_frame(model, frame, style_tensors[active_style])
        else:
            display = frame

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time + 1e-9)
        prev_time = current_time
        cv2.putText(
            display,
            f"FPS: {fps:.1f} | Style: {active_style}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Neural Style Transfer", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            stylize_on = not stylize_on
        elif chr(key) in style_tensors:
            active_style = chr(key)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model = load_hub_model()

    verify_with_tutorial_images(model)

    style_tensors = {
        key: load_and_preprocess(path, target_size=(256, 256))
        for key, path in STYLE_IMAGES.items()
    }
    run_camera(model, style_tensors)
