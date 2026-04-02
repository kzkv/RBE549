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
    """Load MerchData with a 70/30 train/val split, holding out one image per class for testing."""
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

    all_images = []
    all_labels = []
    for images, labels in val_dataset:
        all_images.append(images)
        all_labels.append(labels)
    all_images = tf.concat(all_images, axis=0)
    all_labels = tf.concat(all_labels, axis=0)

    test_indices = []
    val_indices = []
    for class_id in range(NUM_CLASSES):
        class_mask = tf.where(all_labels == class_id)[:, 0].numpy()
        test_indices.append(class_mask[0])
        val_indices.extend(class_mask[1:])

    test_images = tf.gather(all_images, test_indices)
    test_labels = tf.gather(all_labels, test_indices)
    val_images = tf.gather(all_images, val_indices)
    val_labels = tf.gather(all_labels, val_indices)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset, val_dataset, (test_images, test_labels), class_names


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


def test_held_out(model, test_data, class_names, title):
    """Classify 5 held-out test images (one per class) and display in a 3x2 grid."""
    test_images, test_labels = test_data
    predictions = model.predict(test_images)
    predicted_classes = tf.argmax(predictions, axis=1).numpy()
    confidences = tf.reduce_max(predictions, axis=1).numpy()

    fig, axes = plt.subplots(3, 2, figsize=(7, 10))
    fig.suptitle(f"{title} - Test Predictions")

    for i, ax in enumerate(axes.flat):
        if i >= len(test_images):
            ax.axis("off")
            continue
        img = test_images[i].numpy().astype("uint8")
        true_label = class_names[test_labels[i]]
        pred_label = class_names[predicted_classes[i]]
        confidence = confidences[i] * 100
        color = "green" if predicted_classes[i] == test_labels[i] else "red"
        ax.imshow(img)
        ax.set_title(
            f"True: {true_label}\n{pred_label}, {confidence:.1f}%",
            color=color,
            fontsize=9,
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"merch_{title.lower().replace('-', '_')}_test.png", dpi=150)
    plt.show()


def plot_comparison(vgg_history, inception_history):
    """Side-by-side accuracy comparison of VGG-19 and Inception-V3."""
    vgg_val_acc = vgg_history.history["val_accuracy"]
    inc_val_acc = inception_history.history["val_accuracy"]
    epochs = range(1, len(vgg_val_acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, vgg_val_acc, label="VGG-19")
    ax1.plot(epochs, inc_val_acc, label="Inception-V3")
    ax1.set_title("Validation Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    vgg_val_loss = vgg_history.history["val_loss"]
    inc_val_loss = inception_history.history["val_loss"]
    ax2.plot(epochs, vgg_val_loss, label="VGG-19")
    ax2.plot(epochs, inc_val_loss, label="Inception-V3")
    ax2.set_title("Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("merch_comparison.png", dpi=150)
    plt.show()


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
    train_ds, val_ds, test_data_vgg, class_names = load_data((224, 224))
    print(f"Classes: {class_names}")

    vgg_model = build_vgg19_model()
    vgg_model.summary()
    vgg_history = vgg_model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
    plot_training(vgg_history, "VGG-19", "merch_vgg19.png")
    test_held_out(vgg_model, test_data_vgg, class_names, "VGG-19")

    print("\n=== Inception-V3 ===")
    train_ds_inc, val_ds_inc, test_data_inc, _ = load_data((299, 299))

    inception_model = build_inception_model()
    inception_model.summary()
    inception_history = inception_model.fit(
        train_ds_inc, epochs=EPOCHS, validation_data=val_ds_inc
    )
    plot_training(inception_history, "Inception-V3", "merch_inception.png")
    test_held_out(inception_model, test_data_inc, class_names, "Inception-V3")

    plot_comparison(vgg_history, inception_history)
