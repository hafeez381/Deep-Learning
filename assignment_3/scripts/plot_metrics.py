# Usage:
#   python plot_metrics.py --task 1a          # MNIST curves + filter viz
#   python plot_metrics.py --task 1b          # C-MNIST curves + bias comparison
#   python plot_metrics.py --task 2           # STL-10 curves

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.dirname(__file__))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(FIGURES_DIR, exist_ok=True)

# Helpers

def load_metrics(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def plot_curves(metrics: dict, title: str, save_path: str):
    epochs = range(1, len(metrics["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    axes[0].plot(epochs, metrics["train_loss"],
                 label="Train", marker="o", linewidth=2)
    axes[0].plot(epochs, metrics["val_loss"],
                 label="Validation", marker="o", linewidth=2, linestyle="--")
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].plot(epochs, metrics["train_acc"],
                 label="Train", marker="o", linewidth=2)
    axes[1].plot(epochs, metrics["val_acc"],
                 label="Validation", marker="o", linewidth=2, linestyle="--")
    axes[1].set_title("Classification Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


def plot_task_1a():
    metrics_path = os.path.join(RESULTS_DIR, "task1_mnist_metrics.json")
    weights_path = os.path.join(MODELS_DIR,  "custom_cnn_mnist.pth")

    metrics = load_metrics(metrics_path)

    # Training curves
    plot_curves(
        metrics,
        title     = f"Part A — MnistCNN | Test Acc: {metrics['test_acc']:.2f}%",
        save_path = os.path.join(FIGURES_DIR, "task1a_mnist_curves.png")
    )

    # Filter visualisation — requires loading the saved weights
    from models import MnistCNN
    model = MnistCNN()
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()

    filters     = model.conv1[0].weight.data.numpy()  # [8, 1, 3, 3]
    num_filters = filters.shape[0]

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle("Layer 1 Convolutional Filters (3×3)", fontsize=14)

    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            f  = filters[i, 0]
            im = ax.imshow(f, cmap="viridis")
            ax.set_title(f"Filter {i+1}")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "filter_visuals.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


def plot_task_1b():
    metrics_path = os.path.join(RESULTS_DIR, "task1_cmnist_metrics.json")
    metrics      = load_metrics(metrics_path)

    biased_acc   = metrics["biased_test_acc"]
    unbiased_acc = metrics["unbiased_test_acc"]

    # Training curves
    plot_curves(
        metrics,
        title     = (f"Part B — ColoredMnistCNN | "
                     f"Biased: {biased_acc:.2f}%  "
                     f"Unbiased: {unbiased_acc:.2f}%"),
        save_path = os.path.join(FIGURES_DIR, "task1b_cmnist_curves.png")
    )

    # Biased vs unbiased bar chart
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        ["Biased Test", "Unbiased Test"],
        [biased_acc, unbiased_acc],
        color=["steelblue", "tomato"],
        width=0.4, edgecolor="black"
    )
    for bar, val in zip(bars, [biased_acc, unbiased_acc]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.2f}%", ha="center", va="bottom", fontsize=11
        )
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Shortcut Learning:\nBiased vs. Unbiased Test Accuracy")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "task1b_bias_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


def plot_task_2():
    metrics_path = os.path.join(RESULTS_DIR, "task2_stl10_metrics.json")
    metrics      = load_metrics(metrics_path)

    # Rename keys for display so legend reads "Test" not "Validation"
    display_metrics = {
        "train_loss": metrics["train_loss"],
        "train_acc":  metrics["train_acc"],
        "val_loss":   metrics["val_loss"],   # actually test loss
        "val_acc":    metrics["val_acc"],    # actually test acc
    }

    epochs = range(1, len(display_metrics["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Task 2 — ResNet-18 on STL-10 | Test Acc: {metrics['test_acc']:.2f}%",
        fontsize=13, fontweight="bold"
    )

    axes[0].plot(epochs, display_metrics["train_loss"],
                 label="Train", marker="o", linewidth=2)
    axes[0].plot(epochs, display_metrics["val_loss"],
                 label="Test", marker="o", linewidth=2, linestyle="--")
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].plot(epochs, display_metrics["train_acc"],
                 label="Train", marker="o", linewidth=2)
    axes[1].plot(epochs, display_metrics["val_acc"],
                 label="Test", marker="o", linewidth=2, linestyle="--")
    axes[1].set_title("Classification Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "task2_stl10_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate figures from saved JSON metrics."
    )
    parser.add_argument(
        "--task", type=str, required=True,
        choices=["1a", "1b", "2"],
        help="1a → MNIST | 1b → C-MNIST | 2 → STL-10"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.task == "1a":
        plot_task_1a()
    elif args.task == "1b":
        plot_task_1b()
    elif args.task == "2":
        plot_task_2()