import torch
import torch.nn as nn


class CNNBaseline(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, filters: int = 32, dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, filters, kernel_size=7, padding=3),
            nn.BatchNorm1d(filters),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(filters, filters * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(filters * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters * 2, filters * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(filters * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
