# Tom Kazakov
# RBE 549 Lab 11: Transfer Learning with MobileNetV2

import os
import matplotlib.pyplot as plt
import tensorflow as tf

BATCH_SIZE = 32
IMG_SIZE = (160, 160)
IMG_SHAPE = IMG_SIZE + (3,)
AUTOTUNE = tf.data.AUTOTUNE

BASE_LEARNING_RATE = 0.0001
INITIAL_EPOCHS = 10

DATASET_DIR = os.path.join(os.path.dirname(__file__), "data", "cats_and_dogs_filtered")

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

data_augmentation = tf.keras.Sequential(
    [tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomRotation(0.2)]
)


def load_data():
    """Load cats-vs-dogs and return train, validation, and test datasets."""
    train_dir = os.path.join(DATASET_DIR, "train")
    validation_dir = os.path.join(DATASET_DIR, "validation")

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
    )
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        validation_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
    )

    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches // 5)
    validation_dataset = validation_dataset.skip(val_batches // 5)

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset, validation_dataset, test_dataset


def build_model():
    """Build a MobileNetV2 feature-extraction model with a binary classification head."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE, include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=BASE_LEARNING_RATE),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name="accuracy")],
    )
    return model, base_model


if __name__ == "__main__":
    train_ds, val_ds, test_ds = load_data()
    model, base_model = build_model()
    model.summary()

    history = model.fit(train_ds, epochs=INITIAL_EPOCHS, validation_data=val_ds)
