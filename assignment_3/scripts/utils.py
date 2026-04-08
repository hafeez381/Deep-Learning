import torch
import json
import os
from tqdm import tqdm
import numpy as np
import copy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data import TensorDataset

def get_mnist_loaders(data_dir: str, batch_size: int = 64, val_split: float = 0.1, num_workers: int = 0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))   # MNIST channel mean/std
    ])

    full_train = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_set   = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    val_size   = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def get_cmnist_loaders(data_dir: str, batch_size: int = 64,
                       val_split: float = 0.1, num_workers: int = 0):
    """
    Loads C-MNIST .pt files. Each file is a tuple of (images, labels)
    where images are [N, 3, 28, 28] float32 and labels are [N] int64.
    """
    def load_pt(filename):
        images, labels = torch.load(os.path.join(data_dir, filename), weights_only=False)
        return TensorDataset(images, labels)

    full_train = load_pt("train_biased.pt")
    biased_test = load_pt("test_biased.pt")
    unbiased_test = load_pt("test_unbiased.pt")

    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(full_train, [train_size, val_size],generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=True,  num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size,shuffle=False, num_workers=num_workers)
    biased_loader = DataLoader(biased_test, batch_size=batch_size,shuffle=False, num_workers=num_workers)
    unbiased_loader = DataLoader(unbiased_test,batch_size=batch_size,shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, biased_loader, unbiased_loader

def get_stl10_loaders(data_dir: str, batch_size: int = 64, num_workers: int = 0):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),   # ImageNet mean
            std=(0.229, 0.224, 0.225)     # ImageNet std
        )
    ])

    train_set = datasets.STL10(root=data_dir, split='train', download=True, transform=transform)
    test_set  = datasets.STL10(root=data_dir, split='test', download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def save_metrics(metrics: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {path}")


def load_metrics(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def train_one_epoch(model, loader, criterion, optimizer, DEVICE):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()  # Clear old gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels) # Compute loss
        loss.backward()  # Backpropagate
        optimizer.step()  # Update weights

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss/ len(loader)
    train_acc = 100.0 * correct / total

    return train_loss, train_acc


def evaluate(model, loader, criterion, DEVICE):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels) # Compute loss

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    val_loss = running_loss/ len(loader)
    val_acc = 100.0 * correct / total

    return val_loss, val_acc


class EarlyStopping:
    """
    Early stopping regularization to terminate the optimization timeline 
    before gradient descent achieves the absolute global minimum of the training error.
    
    Args:
        patience (int): The parameter 'p'. Number of epochs with no improvement 
                        after which training will be stopped.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        verbose (bool): If True, prints a message for each validation loss improvement.
    """
    def __init__(self, patience = 5, min_delta = 0.0, verbose = False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        """
        Evaluates the current validation loss against the historical best.
        """
        # Case 1: Improvement observed (Loss decreases beyond min_delta)
        if val_loss < self.best_loss - self.min_delta:
            if self.verbose:
                print(f"Validation loss decreased ({self.best_loss:.4f} --> {val_loss:.4f}). Saving optimal state...")
            
            self.best_loss = val_loss
            # Deepcopy detaches the tensors from the active computation graph
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0  # Reset patience counter
            
        # Case 2: Stagnation or degradation observed
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                
            if self.counter >= self.patience:
                self.early_stop = True

    def restore_best_weights(self, model):
        """Restores the parameter state from epoch t-p."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            print("Restored best model weights from epoch t-p.")
        else:
            print("Warning: No best state found to restore.")