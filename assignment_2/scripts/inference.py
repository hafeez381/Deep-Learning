import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Configuration & paths
DATA_DIR = '../data/'
TRAIN_FILE = os.path.join(DATA_DIR, 'quickdraw_train.npz')
TEST_FILE = os.path.join(DATA_DIR, 'quickdraw_test.npz')
WEIGHTS_FILE = '../models/champion_weights.pth'
SUBMISSION_FILE = 'submission.txt'

BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Compute global stats for standardization
print(f"Loading training data to compute global standardization stats...")
train_data = np.load(TRAIN_FILE)
x_train_float = train_data['x_train'].astype(np.float32) / 255.0
global_mean = float(np.mean(x_train_float))
global_std = float(np.std(x_train_float))
num_classes = len(train_data['class_names'])
print(f"Global Mean: {global_mean:.4f} | Global Std: {global_std:.4f}")

# Dataset & dataloader
class TestQuickDrawDataset(Dataset):
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not find test file: {file_path}")
        
        data = np.load(file_path)
        self.x = data['test_images']
        
        # Convert to Float
        self.x = torch.from_numpy(self.x).float()
        
        # Z-Score Standardize using the exact same stats as the training data
        self.x = (self.x / 255.0 - global_mean) / global_std
        
        # Reshape to flatten for MLP
        self.x = self.x.view(-1, 784)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]

print("Initializing Test Dataset and Loader...")
test_dataset = TestQuickDrawDataset(TEST_FILE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Champion model architecture
class ChampionMLP(nn.Module):
    def __init__(self, input_size=784, output_size=num_classes):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 400),
            nn.SELU(),
            nn.AlphaDropout(0.1),
            
            nn.Linear(400, 256),
            nn.SELU(),
            nn.AlphaDropout(0.05),
            
            nn.Linear(256, 128),
            nn.SELU(),
            
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.network(x)

print(f"Loading Champion Model weights from {WEIGHTS_FILE}...")
champion_model = ChampionMLP().to(DEVICE)

# Load the saved state dict
if os.path.exists(WEIGHTS_FILE):
    champion_model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=DEVICE))
    print("Weights loaded successfully.")
else:
    raise FileNotFoundError(f"Weights file not found at {WEIGHTS_FILE}. Please train the model first.")

# Inference loop
def get_predictions(model, loader, device):
    model.eval()
    preds = []
    
    with torch.no_grad():
        for batch in loader:
            X = batch.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            
    return preds

print("Running inference on test set...")
predictions = get_predictions(champion_model, test_loader, DEVICE)

# Save submission file
print(f"Saving predictions to '{SUBMISSION_FILE}'...")
submission_string = ",".join(map(str, predictions))

with open(SUBMISSION_FILE, "w") as f:
    f.write(submission_string)

print(f"Inference complete. Generated {len(predictions)} predictions.")
print(f"Copy the contents of '{SUBMISSION_FILE}' into the leaderboard portal.")