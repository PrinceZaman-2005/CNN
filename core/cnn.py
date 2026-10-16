import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        # Feature extraction
        self.features = nn.Sequential(
            # Input: (batch_size, 1, 8, 8)
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Input: (batch_size, 8, 4, 4)
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Classification
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 2 * 2, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x