import torch
import torch.nn as nn


class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        branch_channels = out_channels // 4
        self.branch1 = nn.Conv1d(in_channels, branch_channels, kernel_size=9, padding=4)
        self.branch2 = nn.Conv1d(in_channels, branch_channels, kernel_size=19, padding=9)
        self.branch3 = nn.Conv1d(in_channels, branch_channels, kernel_size=39, padding=19)
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
        )
        self.bn = nn.BatchNorm1d(branch_channels * 4)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        min_len = min(b1.size(-1), b2.size(-1), b3.size(-1), b4.size(-1))
        b1 = b1[..., :min_len]
        b2 = b2[..., :min_len]
        b3 = b3[..., :min_len]
        b4 = b4[..., :min_len]
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return self.relu(self.bn(out))


class InceptionTimeClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, filters: int = 64, dropout: float = 0.4):
        super().__init__()
        self.block1 = InceptionBlock1D(in_channels, filters)
        self.block2 = InceptionBlock1D(filters, filters)
        self.block3 = InceptionBlock1D(filters, filters)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(filters, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        return self.classifier(x)
