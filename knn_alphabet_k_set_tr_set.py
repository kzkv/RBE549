# Tom Kazakov
# RBE 549 Lab 10: KNN letter recognition, all k values x all train/test splits

import cv2
import numpy as np
import matplotlib.pyplot as plt

DATASET_PATH = "data/letter+recognition/letter-recognition.data"
K_VALUES = range(1, 10)
TRAIN_PERCENTAGES = range(10, 100, 10)


def load_letters(path):
    """Load UCI letter recognition dataset, return features and numeric labels."""
    features = []
    labels = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            labels.append(ord(parts[0]) - ord("A"))
            features.append([int(x) for x in parts[1:]])

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.float32)


def train_and_evaluate(train_data, train_labels, test_data, test_labels, k):
    """Train KNN and return accuracy on test set."""
    knn = cv2.ml.KNearest_create()
    knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
    _, results, _, _ = knn.findNearest(test_data, k)
    correct = np.sum(results.flatten() == test_labels)
    return correct / len(test_labels) * 100


if __name__ == "__main__":
    features, labels = load_letters(DATASET_PATH)
    n_samples = len(labels)

    results = {}
    for train_pct in TRAIN_PERCENTAGES:
        split = n_samples * train_pct // 100
        train_data, test_data = features[:split], features[split:]
        train_labels, test_labels = labels[:split], labels[split:]

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
    plt.title("KNN Letter Recognition -- all k values x all train/test splits")
    plt.xticks(list(K_VALUES))
    plt.legend(title="Train/Test %", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("knn_alphabet_k_set_tr_set.png", dpi=150)
    plt.show()
