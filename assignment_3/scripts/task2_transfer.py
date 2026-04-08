# Usage:
#   python task2_transfer.py --mode train      # fine-tune ResNet-18 on STL-10
#   python task2_transfer.py --mode gradcam    # generate GradCAM visualizations

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image as PILImage

sys.path.append(os.path.dirname(__file__))

from models import FineTunedResNet18
from utils  import get_stl10_loaders, train_one_epoch, evaluate, save_metrics

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "stl10")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

for d in [MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# STL-10 class names in torchvision order
STL10_CLASSES = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

def run_train():
    print(f"Using device: {DEVICE}")

    train_loader, test_loader = get_stl10_loaders(data_dir = DATA_DIR, batch_size = BATCH_SIZE)

    model = FineTunedResNet18(num_classes=10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

    optimizer = optim.Adam(trainable, lr=LEARNING_RATE)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   []
    }

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE)

        # Use test loader for validation since no val split
        val_loss, val_acc = evaluate(
            model, test_loader, criterion, DEVICE)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {val_loss:.4f}  Test Acc: {val_acc:.2f}%")

    # Final test evaluation
    print("\nFinal evaluation on STL-10 test set...")
    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    print(f"\nFinal Test Set Accuracy: {test_acc:.2f}%")

    # Save weights
    weights_path = os.path.join(MODELS_DIR, "resnet18_stl10.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"Weights saved to {weights_path}")

    # Save metrics
    history["test_acc"]  = test_acc
    history["test_loss"] = test_loss
    save_metrics(history, os.path.join(RESULTS_DIR, "task2_stl10_metrics.json"))


class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, input, output):
        self.activations = output
        if output.requires_grad:
            self.activations.retain_grad()

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()

        input_tensor = input_tensor.requires_grad_(True)
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.activations.grad.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam)
        cam     = cam.squeeze().detach().cpu().numpy()

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam, class_idx


def run_gradcam():
    print(f"Using device: {DEVICE}")

    weights_path = os.path.join(MODELS_DIR, "resnet18_stl10.pth")
    model = FineTunedResNet18(num_classes=10).to(DEVICE)
    model.load_state_dict(
        torch.load(weights_path, map_location=DEVICE, weights_only=True)
    )
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    test_set = datasets.STL10(root=DATA_DIR, split='test',
                              download=True, transform=transform)

    # ── Step 1: collect samples under no_grad (no hook yet) ──────────────────
    correct_samples   = []
    incorrect_samples = []

    with torch.no_grad():
        for idx in range(len(test_set)):
            if len(correct_samples) >= 2 and len(incorrect_samples) >= 2:
                break

            image, true_label = test_set[idx]
            input_tensor = image.unsqueeze(0).to(DEVICE)
            pred_label   = model(input_tensor).argmax(dim=1).item()

            if pred_label == true_label and len(correct_samples) < 2:
                correct_samples.append((image, true_label, pred_label))
            elif pred_label != true_label and len(incorrect_samples) < 2:
                incorrect_samples.append((image, true_label, pred_label))

    # ── Step 2: register hook AFTER no_grad block ─────────────────────────────
    target_layer = model.model.layer4[-1]
    gradcam      = GradCAM(model, target_layer)

    # ── Step 3: generate CAMs (gradients enabled) ─────────────────────────────
    all_samples = correct_samples + incorrect_samples
    labels      = ['Correct', 'Correct', 'Incorrect', 'Incorrect']

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("GradCAM Visualizations — Fine-tuned ResNet-18 on STL-10",
                 fontsize=14, fontweight="bold")

    for col, (image, true_label, pred_label) in enumerate(all_samples):
        input_tensor = image.unsqueeze(0).to(DEVICE)
        cam, _       = gradcam.generate(input_tensor, class_idx=pred_label)

        img_np = image.permute(1, 2, 0).numpy()
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)

        cam_resized = np.array(
            PILImage.fromarray((cam * 255).astype(np.uint8)).resize(
                (img_np.shape[1], img_np.shape[0]),
                PILImage.BILINEAR
            )
        ) / 255.0

        result = labels[col]
        title  = (f"{result}\nTrue: {STL10_CLASSES[true_label]}\n"
                  f"Pred: {STL10_CLASSES[pred_label]}")
        color  = "green" if result == "Correct" else "red"

        axes[0, col].imshow(img_np)
        axes[0, col].set_title(title, fontsize=9, color=color, fontweight="bold")
        axes[0, col].axis("off")

        axes[1, col].imshow(img_np)
        axes[1, col].imshow(cam_resized, cmap="jet", alpha=0.45)
        axes[1, col].set_title("GradCAM Heatmap", fontsize=9)
        axes[1, col].axis("off")

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "gradcam_results.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"GradCAM figure saved to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Task 2: Transfer Learning + GradCAM")
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["train", "gradcam"],
        help="train: fine-tune ResNet-18 | gradcam: generate heatmaps"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        run_train()
    elif args.mode == "gradcam":
        run_gradcam()