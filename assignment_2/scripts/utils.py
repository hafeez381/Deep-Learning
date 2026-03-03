import torch
from tqdm import tqdm
import numpy as np
import copy

def train_one_epoch(model, loader, criterion, optimizer, DEVICE, scheduler=None):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()  #Clear old gradients
        outputs = model(inputs)  #Forward pass
        loss = criterion(outputs, labels) #Compute loss
        loss.backward()  #Backpropagate
        optimizer.step()  #Update weights

        # 1CycleR scheduler step after every batch
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss/ len(loader)
    train_acc = 100. * correct / total

    return train_loss, train_acc


def evaluate(model, loader, criterion, DEVICE):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation", leave=False):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(inputs)  #Forward pass
            loss = criterion(outputs, labels) #Compute loss

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    val_loss = running_loss/ len(loader)
    val_acc = 100. * correct / total
    print(f"Validation Acc: {val_acc:.2f}%   Loss: {val_loss:.4f}")

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