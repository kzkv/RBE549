"""KNN digit recognition -- sweep k=[1..9] and train splits [10%..90%], plot all."""

import cv2
import numpy as np
import matplotlib.pyplot as plt

DIGITS_IMAGE = "digits.png"
K_VALUES = range(1, 10)
TRAIN_PERCENTAGES = range(10, 100, 10)
CELL_SIZE = 20
DIGITS = 10
SAMPLES_PER_DIGIT = 500
COLS = 100


def load_digits(path):
    """Load digits.png and return cell array."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load {path}")

    rows_per_digit = SAMPLES_PER_DIGIT // COLS
    cells = [np.hsplit(row, COLS) for row in np.vsplit(img, DIGITS * rows_per_digit)]
    return np.array(cells)


def split_train_test(cells, train_cols):
    """Split cell grid by columns into train/test feature vectors and labels."""
    rows_per_digit = cells.shape[0] // DIGITS
    test_cols = COLS - train_cols

    train_data = (
        cells[:, :train_cols].reshape(-1, CELL_SIZE * CELL_SIZE).astype(np.float32)
    )
    test_data = (
        cells[:, train_cols:].reshape(-1, CELL_SIZE * CELL_SIZE).astype(np.float32)
    )

    train_labels = np.repeat(np.arange(DIGITS), train_cols * rows_per_digit).astype(
        np.float32
    )
    test_labels = np.repeat(np.arange(DIGITS), test_cols * rows_per_digit).astype(
        np.float32
    )

    return train_data, train_labels, test_data, test_labels


def train_and_evaluate(train_data, train_labels, test_data, test_labels, k):
    """Train KNN and return accuracy on test set."""
    knn = cv2.ml.KNearest_create()
    knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
    _, results, _, _ = knn.findNearest(test_data, k)
    correct = np.sum(results.flatten() == test_labels)
    return correct / len(test_labels) * 100


if __name__ == "__main__":
    cells = load_digits(DIGITS_IMAGE)

    results = {}
    for train_pct in TRAIN_PERCENTAGES:
        train_cols = COLS * train_pct // 100
        train_data, train_labels, test_data, test_labels = split_train_test(
            cells, train_cols
        )

        results[train_pct] = []
        for k in K_VALUES:
            acc = train_and_evaluate(
                train_data, train_labels, test_data, test_labels, k
            )
            results[train_pct].append(acc)
            print(f"train={train_pct}%, k={k}: {acc:.2f}%")

    plt.figure(figsize=(10, 6))
    for train_pct in TRAIN_PERCENTAGES:
        label = f"{train_pct}/{100 - train_pct}"
        plt.plot(list(K_VALUES), results[train_pct], marker="o", label=label)

    plt.xlabel("k")
    plt.ylabel("Accuracy (%)")
    plt.title("KNN Digit Recognition -- all k values x all train/test splits")
    plt.xticks(list(K_VALUES))
    plt.legend(title="Train/Test %", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("knn_digits_k_set_tr_set.png", dpi=150)
    plt.show()
