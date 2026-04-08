# scripts/task1_mnist.py
#
# Usage:
#   python task1_mnist.py --mode a       # Standard MNIST
#   python task1_mnist.py --mode b       # Colored MNIST

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(__file__))

from models import MnistCNN, ColoredMnistCNN
from utils import (
    get_mnist_loaders,
    get_cmnist_loaders,
    train_one_epoch,
    evaluate,
    EarlyStopping,
    save_metrics
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

for d in [MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)

EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
PATIENCE = 4


def run_part_a():
    print(" Part A: Standard MNIST")

    train_loader, val_loader, test_loader = get_mnist_loaders(
        data_dir = os.path.join(DATA_DIR, "mnist"),
        batch_size = BATCH_SIZE
    )

    model = MnistCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    early_stopper = EarlyStopping(patience=PATIENCE, verbose=True)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   []
    }

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%")

        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    early_stopper.restore_best_weights(model)

    # Test
    print("\nEvaluating on test set...")
    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    print(f"\nFinal Test Set Accuracy: {test_acc:.2f}%")

    # Save weights
    weights_path = os.path.join(MODELS_DIR, "custom_cnn_mnist.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"Weights saved to {weights_path}")

    # Save metrics
    history["test_acc"]  = test_acc
    history["test_loss"] = test_loss
    save_metrics(history, os.path.join(RESULTS_DIR, "task1_mnist_metrics.json"))


def run_part_b():
    print("Part B: Colored MNIST")

    train_loader, val_loader, biased_loader, unbiased_loader = get_cmnist_loaders(
        data_dir = os.path.join(DATA_DIR, "cmnist"),
        batch_size = BATCH_SIZE
    )

    model = ColoredMnistCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    early_stopper = EarlyStopping(patience=PATIENCE, verbose=True)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   []
    }

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, DEVICE)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%")

        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    early_stopper.restore_best_weights(model)

    # Evaluate on both test sets
    print("\nEvaluating on biased test set...")
    _, biased_acc = evaluate(model, biased_loader, criterion, DEVICE)

    print("\nEvaluating on unbiased test set...")
    _, unbiased_acc = evaluate(model, unbiased_loader, criterion, DEVICE)

    print(f"\n> Biased Test Accuracy : {biased_acc:.2f}%")
    print(f"> Unbiased Test Accuracy : {unbiased_acc:.2f}%")

    # Save weights
    weights_path = os.path.join(MODELS_DIR, "custom_cnn_cmnist.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"Weights saved to {weights_path}")

    # Save metrics
    history["biased_test_acc"] = biased_acc
    history["unbiased_test_acc"] = unbiased_acc
    save_metrics(history, os.path.join(RESULTS_DIR, "task1_cmnist_metrics.json"))


def parse_args():
    parser = argparse.ArgumentParser(description="Task 1 — Custom CNN Training")
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["a", "b"],
        help="a: Standard MNIST | b: Colored MNIST"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Using device: {DEVICE}")

    if args.mode == "a":
        run_part_a()
    elif args.mode == "b":
        run_part_b()