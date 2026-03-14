import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        filters: int = 32,
        lstm_hidden: int = 128,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, filters, kernel_size=7, padding=3),
            nn.BatchNorm1d(filters),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(filters, filters * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(filters * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(
            input_size=filters * 2,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.classifier(x)
