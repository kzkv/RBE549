"""KNN digit recognition on OpenCV digits.png -- vanilla baseline (k=5, 50/50 split)."""

import cv2
import numpy as np

DIGITS_IMAGE = "digits.png"
K = 5
CELL_SIZE = 20
DIGITS = 10
SAMPLES_PER_DIGIT = 500
COLS = 100
TRAIN_COLS = COLS // 2


def load_digits(path):
    """Load digits.png and return feature matrix and label vector."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load {path}")

    rows_per_digit = SAMPLES_PER_DIGIT // COLS
    cells = [np.hsplit(row, COLS) for row in np.vsplit(img, DIGITS * rows_per_digit)]
    cells = np.array(cells)

    features = cells.reshape(-1, CELL_SIZE * CELL_SIZE).astype(np.float32)
    labels = np.repeat(np.arange(DIGITS), SAMPLES_PER_DIGIT).astype(np.float32)
    return features, labels, cells


def split_train_test(features, labels, cells):
    """Split by columns: first half train, second half test."""
    rows_per_digit = SAMPLES_PER_DIGIT // COLS

    train_cells = (
        cells[:, :TRAIN_COLS].reshape(-1, CELL_SIZE * CELL_SIZE).astype(np.float32)
    )
    test_cells = (
        cells[:, TRAIN_COLS:].reshape(-1, CELL_SIZE * CELL_SIZE).astype(np.float32)
    )

    train_labels = np.repeat(np.arange(DIGITS), TRAIN_COLS * rows_per_digit).astype(
        np.float32
    )
    test_labels = np.repeat(
        np.arange(DIGITS), (COLS - TRAIN_COLS) * rows_per_digit
    ).astype(np.float32)

    return train_cells, train_labels, test_cells, test_labels


def train_and_evaluate(train_data, train_labels, test_data, test_labels, k):
    """Train KNN and return accuracy on test set."""
    knn = cv2.ml.KNearest_create()
    knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
    _, results, _, _ = knn.findNearest(test_data, k)
    correct = np.sum(results.flatten() == test_labels)
    return correct / len(test_labels) * 100


if __name__ == "__main__":
    features, labels, cells = load_digits(DIGITS_IMAGE)
    train_data, train_labels, test_data, test_labels = split_train_test(
        features, labels, cells
    )

    accuracy = train_and_evaluate(train_data, train_labels, test_data, test_labels, K)
    print(f"KNN (k={K}, 50/50 split): accuracy = {accuracy:.2f}%")
