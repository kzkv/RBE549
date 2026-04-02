# Tom Kazakov
# RBE 549 Lab 11: Transfer Learning with VGG-19 and Inception-V3

import os
import matplotlib.pyplot as plt
import tensorflow as tf

BATCH_SIZE = 8
VALIDATION_SPLIT = 0.3
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

NUM_CLASSES = 5
LEARNING_RATE = 0.0001
EPOCHS = 20

DATASET_DIR = os.path.join(os.path.dirname(__file__), "data", "MerchData")


def make_augmentation():
    """Create a fresh augmentation pipeline (avoids input shape caching)."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
        ]
    )


def load_data(img_size):
    """Load MerchData with a 70/30 train/val split at the given image size."""
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        batch_size=BATCH_SIZE,
        image_size=img_size,
    )
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        batch_size=BATCH_SIZE,
        image_size=img_size,
    )

    class_names = train_dataset.class_names
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset, val_dataset, class_names


def build_vgg19_model():
    """Build a frozen VGG-19 feature extractor with a classification head."""
    img_shape = (224, 224, 3)
    preprocess = tf.keras.applications.vgg19.preprocess_input

    base_model = tf.keras.applications.VGG19(
        input_shape=img_shape, include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=img_shape)
    x = make_augmentation()(inputs)
    x = preprocess(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def build_inception_model():
    """Build a frozen Inception-V3 feature extractor with a classification head."""
    img_shape = (299, 299, 3)
    preprocess = tf.keras.applications.inception_v3.preprocess_input

    base_model = tf.keras.applications.InceptionV3(
        input_shape=img_shape, include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=img_shape)
    x = make_augmentation()(inputs)
    x = preprocess(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def plot_training(history, title, filename):
    """Plot accuracy and loss curves for a single training run."""
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, acc, label="Training")
    ax1.plot(epochs, val_acc, label="Validation")
    ax1.set_title(f"{title} - Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(epochs, loss, label="Training")
    ax2.plot(epochs, val_loss, label="Validation")
    ax2.set_title(f"{title} - Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


if __name__ == "__main__":
    print("=== VGG-19 ===")
    train_ds, val_ds, class_names = load_data((224, 224))
    print(f"Classes: {class_names}")

    vgg_model = build_vgg19_model()
    vgg_model.summary()
    vgg_history = vgg_model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
    plot_training(vgg_history, "VGG-19", "merch_vgg19.png")

    print("\n=== Inception-V3 ===")
    train_ds, val_ds, class_names = load_data((299, 299))

    inception_model = build_inception_model()
    inception_model.summary()
    inception_history = inception_model.fit(
        train_ds, epochs=EPOCHS, validation_data=val_ds
    )
    plot_training(inception_history, "Inception-V3", "merch_inception.png")
