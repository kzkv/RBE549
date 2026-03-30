# Tom Kazakov
# RBE 549 Lab 10: Real-time digit recognition from camera

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "models/digit_classifier.keras"
CAMERA_INDEX = 0
EPOCHS = 5
BATCH_SIZE = 32
IMG_SIZE = 28
ROI_FRACTION = 0.5
CONFIDENCE_THRESHOLD = 0.5


def build_model():
    """CNN for MNIST digit classification."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10),
        ]
    )


def train_model():
    """Train on MNIST and save."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    model = build_model()
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        verbose=1,
    )
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}, loss: {loss:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model


def preprocess_roi(frame, roi_rect):
    """Extract center ROI, threshold, and resize to 28x28 for MNIST."""
    x, y, w, h = roi_rect
    roi = frame[y : y + h, x : x + w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    resized = cv2.resize(thresh, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return resized.astype("float32") / 255.0


def get_center_roi(frame_h, frame_w):
    """Compute a centered square ROI based on ROI_FRACTION of the shorter side."""
    side = int(min(frame_h, frame_w) * ROI_FRACTION)
    x = (frame_w - side) // 2
    y = (frame_h - side) // 2
    return x, y, side, side


def run_camera(model):
    """Real-time single-digit recognition from webcam."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Camera open. Hold a digit (0-9) in the green box. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        roi_rect = get_center_roi(h, w)
        rx, ry, rw, rh = roi_rect

        digit_img = preprocess_roi(frame, roi_rect)
        input_tensor = digit_img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
        predictions = model(input_tensor, training=False).numpy()[0]
        scores = tf.nn.softmax(predictions).numpy()
        idx = np.argmax(scores)
        confidence = float(scores[idx])

        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)

        if confidence >= CONFIDENCE_THRESHOLD:
            text = f"{idx} ({confidence:.0%})"
            cv2.putText(
                frame,
                text,
                (rx, ry - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Digit Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        print(f"Loading saved model from {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        model = train_model()

    run_camera(model)
