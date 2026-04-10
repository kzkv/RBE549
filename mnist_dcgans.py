# Tom Kazakov
# RBE 549 Lab 12: DCGAN on MNIST

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import glob
import time

import imageio
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 50
NOISE_DIM = 100
NUM_EXAMPLES_TO_GENERATE = 16


def make_generator_model():
    """Build the generator: noise vector to 28x28x1 image."""
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(NOISE_DIM,)))
    model.add(layers.Dense(7 * 7 * 256, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))

    model.add(
        layers.Conv2DTranspose(
            128, (5, 5), strides=(1, 1), padding="same", use_bias=False
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            64, (5, 5), strides=(2, 2), padding="same", use_bias=False
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            1, (5, 5), strides=(2, 2), padding="same", use_bias=False, activation="tanh"
        )
    )
    return model


def make_discriminator_model():
    """Build the discriminator: 28x28x1 image to real/fake logit."""
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(28, 28, 1)))
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    """BCE loss for the discriminator on real and fake batches."""
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


def generator_loss(fake_output):
    """BCE loss for the generator (wants discriminator to say real)."""
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def generate_and_save_images(model, epoch, test_input):
    """Generate a 4x4 grid from fixed noise and save to disk."""
    predictions = model(test_input, training=False)
    plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
        plt.axis("off")
    plt.savefig("mnist_dcgan_epoch_{:04d}.png".format(epoch))
    plt.close()


generator = make_generator_model()
discriminator = make_discriminator_model()
gen_optimizer = tf.keras.optimizers.Adam(1e-4)
disc_optimizer = tf.keras.optimizers.Adam(1e-4)


@tf.function
def train_step(images):
    """One training step: update both generator and discriminator."""
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))


def train(dataset, epochs, seed):
    """Train the GAN and save sample images each epoch."""
    num_batches = tf.data.experimental.cardinality(dataset).numpy()
    for epoch in range(epochs):
        start = time.time()
        for batch_idx, image_batch in enumerate(dataset, 1):
            train_step(image_batch)
            print(
                f"\rEpoch {epoch + 1}/{epochs} - batch {batch_idx}/{num_batches}",
                end="",
                flush=True,
            )
        elapsed = time.time() - start
        print(f"\rEpoch {epoch + 1}/{epochs} - {elapsed:.1f}s" + " " * 20)
        generate_and_save_images(generator, epoch + 1, seed)


if __name__ == "__main__":
    (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = (
        train_images.reshape(train_images.shape[0], 28, 28, 1).astype("float32") - 127.5
    ) / 127.5
    train_dataset = (
        tf.data.Dataset.from_tensor_slices(train_images)
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
    )

    seed = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIM])
    train(train_dataset, EPOCHS, seed)

    anim_file = "mnist_dcgan.gif"
    with imageio.get_writer(anim_file, mode="I") as writer:
        filenames = sorted(glob.glob("mnist_dcgan_epoch_*.png"))
        for filename in filenames:
            image = imageio.v2.imread(filename)
            writer.append_data(image)
