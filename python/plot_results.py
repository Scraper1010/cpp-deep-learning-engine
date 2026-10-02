"""Renders training curves from the C++ trainer's metric history.

Reads checkpoints/history.csv (columns: epoch, train_loss, train_acc,
test_loss, test_acc, seconds) written at the end of every training run, and
saves assets/training_curves.png: loss curves (train + test) on the left,
accuracy curves (train + test) on the right — same style as the PyTorch twin.

Usage:
  python python/plot_results.py
  python python/plot_results.py --history checkpoints/history.csv --output assets/training_curves.png
"""

import argparse
import csv
import os
import sys

# Matplotlib is the only dependency (already in python/requirements.txt).
import matplotlib
matplotlib.use("Agg")  # Headless backend: works without a display server.
import matplotlib.pyplot as plt


def read_history(history_path):
    """Parses history.csv into column lists.

    Takes the CSV path written by src/main.cpp.
    Returns a dict with keys epochs, train_loss, train_acc, test_loss, test_acc.
    Test columns are -1 when evaluation was skipped (--no-eval); those points
    are stored as None and dropped from the test curves. Histories written
    before test_loss existed simply yield a test-loss curve of all None.
    """
    epochs, train_loss, train_acc, test_loss, test_acc = [], [], [], [], []
    with open(history_path, newline="") as history_file:
        column_names = csv.DictReader(history_file).fieldnames or []
        has_test_loss = "test_loss" in column_names
        history_file.seek(0)
        for row in csv.DictReader(history_file):
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            train_acc.append(float(row["train_acc"]))
            # Keep row alignment: None marks "no test point this epoch".
            test_loss_value = float(row["test_loss"]) if has_test_loss else -1.0
            test_loss.append(test_loss_value if test_loss_value >= 0 else None)
            test_acc_value = float(row["test_acc"])
            test_acc.append(test_acc_value if test_acc_value >= 0 else None)
    return {"epochs": epochs, "train_loss": train_loss, "train_acc": train_acc,
            "test_loss": test_loss, "test_acc": test_acc}


def plot_curves(history, output_path):
    """Draws loss + accuracy subplots and saves them to output_path."""
    epochs = history["epochs"]

    # Figure mirrors the twin repo: two panels, markers on every epoch.
    plt.figure(figsize=(10, 4))

    # Left panel: mean cross-entropy per epoch (lower is better).
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train", marker="o")
    # Only plot test points that exist (drops the -1 / --no-eval gaps).
    test_loss_points = [(e, v) for e, v in zip(epochs, history["test_loss"]) if v is not None]
    if test_loss_points:
        loss_epochs, loss_values = zip(*test_loss_points)
        plt.plot(loss_epochs, loss_values, label="Test", marker="s")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Mean cross-entropy")
    plt.grid(True, linestyle=":")
    plt.legend()

    # Right panel: classification accuracy in percent.
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train", marker="o")
    # Only plot test points that exist (drops the -1 / --no-eval gaps).
    test_points = [(e, a) for e, a in zip(epochs, history["test_acc"]) if a is not None]
    if test_points:
        test_epochs, test_values = zip(*test_points)
        plt.plot(test_epochs, test_values, label="Test", marker="s")
    plt.title("Accuracy Curve (%)")
    plt.xlabel("Epoch")
    plt.grid(True, linestyle=":")
    plt.legend()

    plt.tight_layout()
    # Create assets/ on fresh clones so the save never fails.
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[+] Saved training curves to {output_path}")


def main():
    """Parses CLI flags, loads the history, and renders the figure."""
    parser = argparse.ArgumentParser(description="Plot NeuraCPP training curves.")
    parser.add_argument("--history", default=os.path.join("checkpoints", "history.csv"),
                        help="CSV written by the C++ trainer.")
    parser.add_argument("--output", default=os.path.join("assets", "training_curves.png"),
                        help="Where to save the PNG figure.")
    args = parser.parse_args()

    if not os.path.exists(args.history):
        print(f"Error: '{args.history}' not found.", file=sys.stderr)
        print("Train first: ./build/neuracpp --epochs 5 --output checkpoints/model.bin", file=sys.stderr)
        raise SystemExit(1)

    history = read_history(args.history)
    if not history["epochs"]:
        print(f"Error: '{args.history}' holds no epochs.", file=sys.stderr)
        raise SystemExit(1)

    plot_curves(history, args.output)


if __name__ == "__main__":
    main()
