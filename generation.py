# Tom Kazakov
# RBE 549 Lab 12: Convolutional Variational Autoencoder on MNIST

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import glob
import time

import imageio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

LATENT_DIM = 2
EPOCHS = 10
BATCH_SIZE = 32
TRAIN_SIZE = 60000
TEST_SIZE = 10000
NUM_EXAMPLES_TO_GENERATE = 16


class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(32, 3, strides=(2, 2), activation="relu"),
                tf.keras.layers.Conv2D(64, 3, strides=(2, 2), activation="relu"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(7 * 7 * 32, activation="relu"),
                tf.keras.layers.Reshape((7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(
                    64, 3, strides=2, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv2DTranspose(
                    32, 3, strides=2, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv2DTranspose(1, 3, strides=1, padding="same"),
            ]
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            return tf.sigmoid(logits)
        return logits


def preprocess_images(images):
    """Normalize and binarize images at 0.5 threshold."""
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.0
    return np.where(images > 0.5, 1.0, 0.0).astype("float32")


def log_normal_pdf(sample, mean, logvar, raxis=1):
    """Compute log probability under a normal distribution."""
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis,
    )


def compute_loss(model, x):
    """Compute the negative ELBO loss."""
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0.0, 0.0)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x, optimizer):
    """Execute one training step and apply gradients."""
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def generate_and_save_images(model, epoch, test_sample):
    """Encode test samples, then decode and save as a 4x4 grid."""
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap="gray")
        plt.axis("off")
    plt.savefig("image_at_epoch_{:04d}.png".format(epoch))
    plt.close()


def plot_latent_images(model, n, digit_size=28):
    """Decode a grid of points sampled uniformly from the latent space."""
    # np.linspace replaces tfp.distributions.Normal.quantile to avoid the tensorflow-probability dependency
    grid_x = np.linspace(-2, 2, n)
    grid_y = np.linspace(-2, 2, n)
    image = np.zeros((digit_size * n, digit_size * n))
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z = np.array([[xi, yi]])
            x_decoded = model.sample(z)
            digit = tf.reshape(x_decoded[0], (digit_size, digit_size))
            image[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit.numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap="Greys_r")
    plt.axis("off")
    plt.savefig("cvae_latent_space.png")
    plt.show()


if __name__ == "__main__":
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)

    train_dataset = (
        tf.data.Dataset.from_tensor_slices(train_images)
        .shuffle(TRAIN_SIZE)
        .batch(BATCH_SIZE)
    )
    test_dataset = (
        tf.data.Dataset.from_tensor_slices(test_images)
        .shuffle(TEST_SIZE)
        .batch(BATCH_SIZE)
    )

    model = CVAE(LATENT_DIM)
    optimizer = tf.keras.optimizers.Adam(1e-4)

    for test_batch in test_dataset.take(1):
        test_sample = test_batch[:NUM_EXAMPLES_TO_GENERATE]

    generate_and_save_images(model, 0, test_sample)

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        for train_x in train_dataset:
            train_step(model, train_x, optimizer)
        end_time = time.time()

        loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
            loss(compute_loss(model, test_x))
        elbo = -loss.result()
        print(
            f"Epoch: {epoch}, Test set ELBO: {elbo:.2f}, "
            f"time: {end_time - start_time:.2f}s"
        )
        generate_and_save_images(model, epoch, test_sample)

    anim_file = "cvae.gif"
    with imageio.get_writer(anim_file, mode="I") as writer:
        filenames = sorted(glob.glob("image_at_epoch_*.png"))
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    plot_latent_images(model, 20)
