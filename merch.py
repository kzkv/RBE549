# Tom Kazakov
# RBE 549 Lab 11: Transfer Learning with VGG-19 and Inception-V3

import os
import matplotlib.pyplot as plt
import tensorflow as tf

BATCH_SIZE = 8
VALIDATION_SPLIT = 0.3
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

DATASET_DIR = os.path.join(os.path.dirname(__file__), "data", "MerchData")


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


if __name__ == "__main__":
    train_ds, val_ds, class_names = load_data((224, 224))
    print(f"Classes: {class_names}")
    print(f"Train batches: {tf.data.experimental.cardinality(train_ds)}")
    print(f"Val batches:   {tf.data.experimental.cardinality(val_ds)}")

    for images, labels in train_ds.take(1):
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")
