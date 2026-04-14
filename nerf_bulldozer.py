# Tom Kazakov
# RBE 549 Lab 13: NeRF on Bulldozer Dataset

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KERAS_BACKEND"] = "tensorflow"

import glob

import imageio.v2 as imageio
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers
from tqdm import tqdm

tf.random.set_seed(42)

DATA_URL = (
    "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz"
)
BATCH_SIZE = 5
NUM_SAMPLES = 32
POS_ENCODE_DIMS = 16
EPOCHS = 20
NEAR = 2.0
FAR = 6.0
DENSE_UNITS = 64
NUM_LAYERS = 8
TRAIN_SPLIT = 0.8


def load_data():
    """Load the tiny NeRF bulldozer dataset and split into train/val."""
    data = np.load(keras.utils.get_file(origin=DATA_URL))
    images = data["images"]
    poses = data["poses"]
    focal = float(data["focal"])

    num_images = images.shape[0]
    split = int(num_images * TRAIN_SPLIT)

    train_images = images[:split]
    val_images = images[split:]
    train_poses = poses[:split]
    val_poses = poses[split:]

    return train_images, val_images, train_poses, val_poses, focal


def encode_position(x):
    """Map input coordinates to higher-dimensional Fourier feature space."""
    positions = [x]
    for i in range(POS_ENCODE_DIMS):
        for fn in [tf.sin, tf.cos]:
            positions.append(fn(2.0**i * x))
    return tf.concat(positions, axis=-1)


def get_rays(height, width, focal, pose):
    """Compute ray origins and direction vectors for every pixel."""
    i, j = tf.meshgrid(
        tf.range(width, dtype=tf.float32),
        tf.range(height, dtype=tf.float32),
        indexing="xy",
    )
    directions = tf.stack(
        [(i - width * 0.5) / focal, -(j - height * 0.5) / focal, -tf.ones_like(i)],
        axis=-1,
    )

    camera_matrix = pose[:3, :3]
    ray_directions = tf.reduce_sum(directions[..., None, :] * camera_matrix, axis=-1)
    ray_origins = tf.broadcast_to(pose[:3, -1], tf.shape(ray_directions))
    return ray_origins, ray_directions


def render_flat_rays(ray_origins, ray_directions, near, far, num_samples, rand=False):
    """Sample points along rays, positional-encode them, and flatten."""
    t_vals = tf.linspace(near, far, num_samples)
    if rand:
        shape = list(ray_origins.shape[:-1]) + [num_samples]
        noise = tf.random.uniform(shape=shape) * (far - near) / num_samples
        t_vals = t_vals + noise

    rays = ray_origins[..., None, :] + (
        ray_directions[..., None, :] * t_vals[..., None]
    )
    rays_flat = tf.reshape(rays, [-1, 3])
    rays_flat = encode_position(rays_flat)
    return rays_flat, t_vals


def render_rgb_depth(model, rays_flat, t_vals, rand=True, train=True):
    """Apply the volume rendering equation to produce an RGB image and depth map."""
    if train:
        predictions = model(rays_flat)
    else:
        predictions = model.predict(rays_flat)
    predictions = tf.reshape(predictions, shape=(BATCH_SIZE, H, W, NUM_SAMPLES, 4))

    rgb = tf.sigmoid(predictions[..., :-1])
    sigma_a = tf.nn.relu(predictions[..., -1])

    delta = t_vals[..., 1:] - t_vals[..., :-1]
    if rand:
        delta = tf.concat(
            [delta, tf.broadcast_to([1e10], shape=(BATCH_SIZE, H, W, 1))], axis=-1
        )
        alpha = 1.0 - tf.exp(-sigma_a * delta)
    else:
        delta = tf.concat(
            [delta, tf.broadcast_to([1e10], shape=(BATCH_SIZE, 1))], axis=-1
        )
        alpha = 1.0 - tf.exp(-sigma_a * delta[:, None, None, :])

    exp_term = 1.0 - alpha
    transmittance = tf.math.cumprod(exp_term + 1e-10, axis=-1, exclusive=True)
    weights = alpha * transmittance
    rgb = tf.reduce_sum(weights[..., None] * rgb, axis=-2)

    if rand:
        depth_map = tf.reduce_sum(weights * t_vals, axis=-1)
    else:
        depth_map = tf.reduce_sum(weights * t_vals[:, None, None], axis=-1)
    return rgb, depth_map


def get_nerf_model(num_layers, num_pos):
    """Build the NeRF MLP with a skip connection at the midpoint."""
    inputs = keras.Input(shape=(num_pos, 2 * 3 * POS_ENCODE_DIMS + 3))
    x = inputs
    for i in range(num_layers):
        x = layers.Dense(units=DENSE_UNITS, activation="relu")(x)
        if i % 4 == 0 and i > 0:
            x = layers.concatenate([x, inputs], axis=-1)
    outputs = layers.Dense(units=4)(x)
    return keras.Model(inputs=inputs, outputs=outputs)


class NeRF(keras.Model):
    """Wraps the NeRF MLP with a custom training loop for volumetric rendering."""

    def __init__(self, nerf_model):
        super().__init__()
        self.nerf_model = nerf_model

    def compile(self, optimizer, loss_fn):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.psnr_metric = keras.metrics.Mean(name="psnr")

    def train_step(self, inputs):
        images, rays = inputs
        rays_flat, t_vals = rays

        with tf.GradientTape() as tape:
            rgb, _ = render_rgb_depth(
                model=self.nerf_model,
                rays_flat=rays_flat,
                t_vals=t_vals,
                rand=True,
            )
            loss = self.loss_fn(images, rgb)

        gradients = tape.gradient(loss, self.nerf_model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.nerf_model.trainable_variables)
        )

        psnr = tf.image.psnr(images, rgb, max_val=1.0)
        self.loss_tracker.update_state(loss)
        self.psnr_metric.update_state(psnr)
        return {"loss": self.loss_tracker.result(), "psnr": self.psnr_metric.result()}

    def test_step(self, inputs):
        images, rays = inputs
        rays_flat, t_vals = rays

        rgb, _ = render_rgb_depth(
            model=self.nerf_model,
            rays_flat=rays_flat,
            t_vals=t_vals,
            rand=True,
        )
        loss = self.loss_fn(images, rgb)

        psnr = tf.image.psnr(images, rgb, max_val=1.0)
        self.loss_tracker.update_state(loss)
        self.psnr_metric.update_state(psnr)
        return {"loss": self.loss_tracker.result(), "psnr": self.psnr_metric.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.psnr_metric]
