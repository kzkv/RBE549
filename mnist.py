# Tom Kazakov
# RBE 549 Lab 10: MNIST classification with TF/Keras (sequential, functional, subclassing)

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import tensorflow as tf

EPOCHS_OPTIMIZER_SWEEP = 5
EPOCHS_DEEP_MODEL = 10
BATCH_SIZE = 32

OPTIMIZERS = {
    "Adam": tf.keras.optimizers.Adam,
    "SGD": tf.keras.optimizers.SGD,
    "RMSProp": tf.keras.optimizers.RMSprop,
    "Adagrad": tf.keras.optimizers.Adagrad,
    "Nadam": tf.keras.optimizers.Nadam,
    "Adamax": tf.keras.optimizers.Adamax,
}


def load_mnist():
    """Load and normalize MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    return (x_train, y_train), (x_test, y_test)


def build_sequential():
    """Baseline sequential model from the beginner quickstart."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10),
        ]
    )


def run_optimizer_sweep(x_train, y_train, x_test, y_test):
    """Train the baseline model with each optimizer, return histories."""
    results = {}
    for name, optimizer_cls in OPTIMIZERS.items():
        print(f"\n--- Optimizer: {name} ---")
        model = build_sequential()
        model.compile(
            optimizer=optimizer_cls(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        history = model.fit(
            x_train,
            y_train,
            epochs=EPOCHS_OPTIMIZER_SWEEP,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            verbose=1,
        )
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"{name}: test accuracy = {acc:.4f}, test loss = {loss:.4f}")
        results[name] = history.history
    return results


def plot_optimizer_sweep(results):
    """Plot accuracy and loss curves for each optimizer."""
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))

    for name, hist in results.items():
        epochs = range(1, len(hist["val_accuracy"]) + 1)
        ax_acc.plot(epochs, hist["val_accuracy"], marker="o", label=name)
        ax_loss.plot(epochs, hist["val_loss"], marker="o", label=name)

    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Validation Accuracy")
    ax_acc.set_title("Optimizer Comparison -- Accuracy")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Validation Loss")
    ax_loss.set_title("Optimizer Comparison -- Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("mnist_optimizer_sweep.png", dpi=150)
    plt.show()


def build_functional():
    """Baseline model using the functional API."""
    inputs = tf.keras.Input(shape=(28, 28))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(10)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def run_functional_model(x_train, y_train, x_test, y_test):
    """Train the functional model."""
    print("\n--- Functional Model ---")
    model = build_functional()
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(
        x_train,
        y_train,
        epochs=EPOCHS_OPTIMIZER_SWEEP,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        verbose=1,
    )
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Functional model: test accuracy = {acc:.4f}, test loss = {loss:.4f}")
    return model


class MNISTModel(tf.keras.Model):
    """Custom model using the subclassing API from the advanced quickstart."""

    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(10)

    def call(self, x, training=False):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.dense2(x)


def run_subclassing_model(x_train, y_train, x_test, y_test):
    """Train the subclassing model following the advanced tutorial."""
    print("\n--- Subclassing Model (Advanced Tutorial) ---")
    model = MNISTModel()
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(
        x_train,
        y_train,
        epochs=EPOCHS_OPTIMIZER_SWEEP,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        verbose=1,
    )
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Subclassing model: test accuracy = {acc:.4f}, test loss = {loss:.4f}")
    return model


def build_deep_sequential():
    """Deeper sequential model with additional layers."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10),
        ]
    )


def run_deep_model(x_train, y_train, x_test, y_test):
    """Train the deeper model and return history."""
    print("\n--- Deep Sequential Model ---")
    model = build_deep_sequential()
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.summary()

    log_dir = "logs/mnist_deep"
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    tf.keras.utils.plot_model(
        model, to_file="mnist_deep_model.png", show_shapes=True, show_layer_names=True
    )
    print("Model graph saved to mnist_deep_model.png")
    print(f"TensorBoard logs: {log_dir}  (run: tensorboard --logdir {log_dir})")

    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS_DEEP_MODEL,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        callbacks=[tensorboard_cb],
        verbose=1,
    )
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Deep model: test accuracy = {acc:.4f}, test loss = {loss:.4f}")
    return history.history


def plot_deep_model(history):
    """Plot accuracy and loss for the deep model."""
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["val_accuracy"]) + 1)

    ax_acc.plot(epochs, history["accuracy"], label="Train")
    ax_acc.plot(epochs, history["val_accuracy"], label="Validation")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Deep Model -- Accuracy")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    ax_loss.plot(epochs, history["loss"], label="Train")
    ax_loss.plot(epochs, history["val_loss"], label="Validation")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Deep Model -- Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("mnist_deep_model_results.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_mnist()

    sweep_results = run_optimizer_sweep(x_train, y_train, x_test, y_test)
    plot_optimizer_sweep(sweep_results)

    run_functional_model(x_train, y_train, x_test, y_test)

    run_subclassing_model(x_train, y_train, x_test, y_test)

    deep_history = run_deep_model(x_train, y_train, x_test, y_test)
    plot_deep_model(deep_history)
