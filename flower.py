# Tom Kazakov
# RBE 549 Lab 10: Flower classification with real-time camera inference

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pathlib

import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = (180, 180)
BATCH_SIZE = 32
EPOCHS = 15
MODEL_PATH = "models/flower_classifier.keras"
CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
CAMERA_INDEX = 0


def build_model():
    """Build the CNN from the TF flowers tutorial with data augmentation."""
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", input_shape=(*IMG_SIZE, 3)),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ]
    )

    return tf.keras.Sequential(
        [
            data_augmentation,
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(len(CLASS_NAMES)),
        ]
    )


def train_model():
    """Download flower dataset, train CNN, save to disk."""
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file("flower_photos", origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir) / "flower_photos"

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    model = build_model()
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"Validation accuracy: {acc:.4f}, loss: {loss:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model


def classify_frame(model, frame):
    """Run inference on a single BGR frame, return class name and confidence."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE)
    input_tensor = np.expand_dims(resized, axis=0).astype(np.float32)
    predictions = model(input_tensor, training=False).numpy()[0]
    scores = tf.nn.softmax(predictions).numpy()
    idx = np.argmax(scores)
    return CLASS_NAMES[idx], float(scores[idx])


def run_camera(model):
    """Real-time flower classification from webcam."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Camera open. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label, confidence = classify_frame(model, frame)
        text = f"{label}: {confidence:.1%}"
        cv2.putText(
            frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2
        )

        cv2.imshow("Flower Classifier", frame)
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
