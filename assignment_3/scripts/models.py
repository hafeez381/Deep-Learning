import torch
import torch.nn as nn
import torchvision.models as tv_models

class MnistCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x

class ColoredMnistCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            # in_channels=3 for RGB input
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x

class FineTunedResNet18(nn.Module):
    """
    Pre-trained ResNet-18 with frozen backbone and a new 10-class head.
    Only the final FC layer is trainable.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)

        for param in self.model.parameters():
            param.requires_grad = False

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


if __name__ == "__main__":
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    mnist_model = MnistCNN()
    cmnist_model = ColoredMnistCNN()

    print(f"MnistCNN params: {count_params(mnist_model):,}")
    print(f"ColoredMnistCNN params: {count_params(cmnist_model):,}")