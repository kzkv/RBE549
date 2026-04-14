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
