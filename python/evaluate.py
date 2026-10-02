"""Evaluates the trained checkpoint on MNIST test images.

Loads checkpoints/model.bin with the NumPy twin (neuracpp_predict), parses
the IDX test archives directly (no torch needed), classifies a subset of the
test split, prints accuracy, and saves assets/confusion_matrix.png: a 10x10
grid of true digit (rows) vs predicted digit (columns) in row-normalized %.

The NumPy forward pass is intentionally unoptimized (mirrors the C++ loops),
so the default is 1000 test samples (~a few minutes). Pass --full for all
10000 test images when you have time.

Usage:
  python python/evaluate.py
  python python/evaluate.py --samples 2000 --output assets/confusion_matrix.png
  python python/evaluate.py --full
"""

import argparse
import os
import struct
import sys

import matplotlib
matplotlib.use("Agg")  # Headless backend: works without a display server.
import matplotlib.pyplot as plt
import numpy as np

from neuracpp_predict import load_model, predict_proba


def read_idx_images(images_path):
    """Parses an IDX3 image archive into (count, 28, 28) float32 in [0, 1]."""
    with open(images_path, "rb") as images_file:
        # Header: magic 2051, image count, rows, cols (all big-endian u32).
        magic, image_count, row_count, col_count = struct.unpack(">4I", images_file.read(16))
        if magic != 2051:
            raise ValueError(f"Bad image magic {magic} in {images_path}")
        pixel_bytes = images_file.read(image_count * row_count * col_count)
    pixels = np.frombuffer(pixel_bytes, dtype=np.uint8).astype(np.float32) / 255.0
    return pixels.reshape(image_count, row_count, col_count)


def read_idx_labels(labels_path):
    """Parses an IDX1 label archive into a (count,) int array of digits."""
    with open(labels_path, "rb") as labels_file:
        # Header: magic 2049, label count (both big-endian u32).
        magic, label_count = struct.unpack(">2I", labels_file.read(8))
        if magic != 2049:
            raise ValueError(f"Bad label magic {magic} in {labels_path}")
        label_bytes = labels_file.read(label_count)
    return np.frombuffer(label_bytes, dtype=np.uint8).astype(int)


def confusion_matrix(true_digits, pred_digits, class_count=10):
    """Counts predictions into a (classes, classes) true-row / pred-column grid."""
    grid = np.zeros((class_count, class_count), dtype=int)
    for true_digit, pred_digit in zip(true_digits, pred_digits):
        grid[true_digit, pred_digit] += 1
    return grid


def plot_confusion(grid, output_path, sample_count):
    """Renders the row-normalized confusion grid and saves it as PNG."""
    # Row-normalize to % so rare confusions stay visible next to the diagonal.
    row_totals = grid.sum(axis=1, keepdims=True).clip(min=1)
    grid_percent = 100.0 * grid / row_totals

    plt.figure(figsize=(7, 6))
    heatmap = plt.imshow(grid_percent, vmin=0, vmax=100, cmap="Blues")
    plt.colorbar(heatmap, label="Row %")
    plt.title(f"Confusion Matrix ({sample_count} test samples)")
    plt.xlabel("Predicted digit")
    plt.ylabel("True digit")
    plt.xticks(range(10))
    plt.yticks(range(10))

    # Annotate every cell: white text on dark cells, black on light ones.
    for true_digit in range(10):
        for pred_digit in range(10):
            cell_value = grid_percent[true_digit, pred_digit]
            text_color = "white" if cell_value > 50 else "black"
            plt.text(pred_digit, true_digit, f"{cell_value:.0f}",
                     ha="center", va="center", color=text_color, fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[+] Saved confusion matrix to {output_path}")


def main():
    """Parses flags, classifies test images, prints accuracy, saves the figure."""
    parser = argparse.ArgumentParser(description="Evaluate NeuraCPP on MNIST test data.")
    parser.add_argument("--model", default=os.path.join("checkpoints", "model.bin"),
                        help="Checkpoint written by the C++ trainer.")
    parser.add_argument("--images", default=os.path.join("data", "t10k-images-idx3-ubyte"))
    parser.add_argument("--labels", default=os.path.join("data", "t10k-labels-idx1-ubyte"))
    parser.add_argument("--samples", type=int, default=1000,
                        help="Test images to classify (default 1000 for speed).")
    parser.add_argument("--full", action="store_true",
                        help="Classify all 10000 test images (slow, ~10x the default).")
    parser.add_argument("--output", default=os.path.join("assets", "confusion_matrix.png"))
    args = parser.parse_args()

    for required_path in (args.model, args.images, args.labels):
        if not os.path.exists(required_path):
            print(f"Error: '{required_path}' not found.", file=sys.stderr)
            raise SystemExit(1)

    # Load the trained weights once, reuse for every test image.
    network_params = load_model(args.model)
    print(f"[+] Loaded weights from {args.model}")

    test_images = read_idx_images(args.images)
    test_labels = read_idx_labels(args.labels)
    sample_count = len(test_images) if args.full else min(args.samples, len(test_images))
    print(f"[*] Classifying {sample_count} test images with NumPy (one dot = 100)...")

    predictions = []
    for sample_index in range(sample_count):
        probabilities = predict_proba(test_images[sample_index], network_params)
        predictions.append(int(np.argmax(probabilities)))
        # Progress heartbeat: the loop is slow, never look frozen.
        if (sample_index + 1) % 100 == 0:
            print(f"  ... {sample_index + 1}/{sample_count}")
    predictions = np.array(predictions)
    true_subset = test_labels[:sample_count]

    accuracy = 100.0 * (predictions == true_subset).mean()
    print(f"[+] Test accuracy on {sample_count} samples: {accuracy:.2f}%")

    grid = confusion_matrix(true_subset, predictions)
    plot_confusion(grid, args.output, sample_count)


if __name__ == "__main__":
    main()
