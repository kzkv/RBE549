# Tom Kazakov
# RBE 549 Lab 12: Denoising autoencoder on Fashion MNIST

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import tensorflow as tf

NOISE_FACTOR = 0.2
EPOCHS = 10


class Denoise(tf.keras.Model):
    """Convolutional autoencoder trained to remove Gaussian noise."""

    def __init__(self):
        super().__init__()
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(
                    16, (3, 3), activation="relu", padding="same", strides=2
                ),
                tf.keras.layers.Conv2D(
                    8, (3, 3), activation="relu", padding="same", strides=2
                ),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2DTranspose(
                    8, (3, 3), strides=2, activation="relu", padding="same"
                ),
                tf.keras.layers.Conv2DTranspose(
                    16, (3, 3), strides=2, activation="relu", padding="same"
                ),
                tf.keras.layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same"),
            ]
        )

    def call(self, x):
        return self.decoder(self.encoder(x))


def load_data():
    """Load and normalize Fashion MNIST, expand dims for conv layers."""
    (x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    return x_train, x_test


def add_noise(images, noise_factor):
    """Add Gaussian noise and clip to [0, 1]."""
    noisy = images + noise_factor * tf.random.normal(shape=images.shape)
    return tf.clip_by_value(noisy, 0.0, 1.0)


def plot_denoised(model, test_images, noisy_images, n=10):
    """Show original, noisy, and denoised images in three rows."""
    denoised = model(noisy_images[:n])
    fig, axes = plt.subplots(3, n, figsize=(20, 6))
    for i in range(n):
        axes[0, i].imshow(tf.squeeze(test_images[i]), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(tf.squeeze(noisy_images[i]), cmap="gray")
        axes[1, i].axis("off")
        axes[2, i].imshow(tf.squeeze(denoised[i]), cmap="gray")
        axes[2, i].axis("off")
    axes[0, 0].set_title("Original")
    axes[1, 0].set_title("Noisy")
    axes[2, 0].set_title("Denoised")
    plt.tight_layout()
    plt.savefig("image_denoising.png")
    plt.show()


if __name__ == "__main__":
    x_train, x_test = load_data()
    x_train_noisy = add_noise(x_train, NOISE_FACTOR)
    x_test_noisy = add_noise(x_test, NOISE_FACTOR)

    denoiser = Denoise()
    denoiser.compile(optimizer="adam", loss="mse")
    denoiser.fit(
        x_train_noisy,
        x_train,
        epochs=EPOCHS,
        shuffle=True,
        validation_data=(x_test_noisy, x_test),
    )

    plot_denoised(denoiser, x_test, x_test_noisy)
