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
EPOCHS = 2
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
    predictions = model(rays_flat, training=train)
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


def build_dataset(images, poses, height, width, focal, shuffle):
    """Build a batched tf.data.Dataset yielding (image, (rays_flat, t_vals))."""

    def map_fn(pose):
        ray_origins, ray_directions = get_rays(height, width, focal, pose)
        return render_flat_rays(
            ray_origins,
            ray_directions,
            near=NEAR,
            far=FAR,
            num_samples=NUM_SAMPLES,
            rand=True,
        )

    img_ds = tf.data.Dataset.from_tensor_slices(images)
    pose_ds = tf.data.Dataset.from_tensor_slices(poses)
    ray_ds = pose_ds.map(map_fn).cache()
    ds = tf.data.Dataset.zip((img_ds, ray_ds))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(images), reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    num_batches = len(images) // BATCH_SIZE
    ds = ds.apply(tf.data.experimental.assert_cardinality(num_batches))
    return ds.prefetch(tf.data.AUTOTUNE)


loss_list = []


class TrainMonitor(keras.callbacks.Callback):
    """Save predicted RGB, depth, and loss-curve figures at each epoch end."""

    def __init__(self, test_rays_flat, test_t_vals, output_dir="nerf_bulldozer_epochs"):
        super().__init__()
        self.test_rays_flat = test_rays_flat
        self.test_t_vals = test_t_vals
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def on_epoch_begin(self, epoch, logs=None):
        print(f"\n[epoch {epoch + 1}/{EPOCHS}] starting", flush=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss", float("nan"))
        psnr = logs.get("psnr", float("nan"))
        val_loss = logs.get("val_loss", float("nan"))
        val_psnr = logs.get("val_psnr", float("nan"))
        print(
            f"[epoch {epoch + 1}/{EPOCHS}] "
            f"loss={loss:.4f} psnr={psnr:.2f} "
            f"val_loss={val_loss:.4f} val_psnr={val_psnr:.2f}",
            flush=True,
        )
        loss_list.append(loss)
        rgb, depth = render_rgb_depth(
            model=self.model.nerf_model,
            rays_flat=self.test_rays_flat,
            t_vals=self.test_t_vals,
            rand=True,
            train=False,
        )

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
        ax[0].imshow(keras.utils.array_to_img(rgb[0]))
        ax[0].set_title(f"Predicted Image: {epoch:03d}")
        ax[1].imshow(keras.utils.array_to_img(depth[0, ..., None]))
        ax[1].set_title(f"Depth Map: {epoch:03d}")
        ax[2].plot(loss_list)
        ax[2].set_xticks(np.arange(0, EPOCHS + 1, 5.0))
        ax[2].set_title(f"Loss Plot: {epoch:03d}")

        fig.savefig(f"{self.output_dir}/{epoch:03d}.png")
        plt.close(fig)


if __name__ == "__main__":
    train_images, val_images, train_poses, val_poses, focal = load_data()
    H, W = train_images.shape[1:3]

    train_ds = build_dataset(train_images, train_poses, H, W, focal, shuffle=True)
    val_ds = build_dataset(val_images, val_poses, H, W, focal, shuffle=False)

    test_imgs, test_rays = next(iter(train_ds))
    test_rays_flat, test_t_vals = test_rays

    nerf_model = get_nerf_model(num_layers=NUM_LAYERS, num_pos=H * W * NUM_SAMPLES)
    model = NeRF(nerf_model)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss_fn=keras.losses.MeanSquaredError(),
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[TrainMonitor(test_rays_flat, test_t_vals)],
    )
