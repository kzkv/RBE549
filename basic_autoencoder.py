# Tom Kazakov
# RBE 549 Lab 12: Basic autoencoder on MNIST

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import tensorflow as tf

LATENT_DIM = 64
EPOCHS = 10
BATCH_SIZE = 256


class Autoencoder(tf.keras.Model):
    """Dense autoencoder with a single-layer encoder and decoder."""

    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(latent_dim, activation="relu"),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(784, activation="sigmoid"),
                tf.keras.layers.Reshape((28, 28)),
            ]
        )

    def call(self, x):
        return self.decoder(self.encoder(x))


def load_data():
    """Load and normalize Fashion MNIST dataset to [0, 1]."""
    (x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    return x_train, x_test


def plot_reconstructions(model, test_images, n=10):
    """Show original and reconstructed images side by side."""
    reconstructed = model(test_images[:n])
    fig, axes = plt.subplots(2, n, figsize=(20, 4))
    for i in range(n):
        axes[0, i].imshow(test_images[i], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(reconstructed[i], cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_title("Original")
    axes[1, 0].set_title("Reconstructed")
    plt.tight_layout()
    plt.savefig("basic_autoencoder.png")
    plt.show()


if __name__ == "__main__":
    x_train, x_test = load_data()

    autoencoder = Autoencoder(LATENT_DIM)
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(
        x_train,
        x_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_data=(x_test, x_test),
    )

    plot_reconstructions(autoencoder, x_test)
