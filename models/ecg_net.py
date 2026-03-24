import torch
import torch.nn as nn

class ECGNet1D(nn.Module):
    def __init__(self, n_classes=5, dropout=0.3):
        super(ECGNet1D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.15),

            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.15),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        z = self.features(x).squeeze(-1)
        return self.classifier(z)

class ECGNet1D_Narrow(nn.Module):
    def __init__(self, c1, c2, c3, c4, c5, fc1, n_classes=5, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=7, padding=3),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.15),
            nn.Conv1d(c2, c3, kernel_size=5, padding=2),
            nn.BatchNorm1d(c3),
            nn.ReLU(inplace=True),
            nn.Conv1d(c3, c4, kernel_size=3, padding=1),
            nn.BatchNorm1d(c4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.15),
            nn.Conv1d(c4, c5, kernel_size=3, padding=1),
            nn.BatchNorm1d(c5),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(c5, fc1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(fc1, n_classes),
        )
    def forward(self, x):
        z = self.features(x).squeeze(-1)
        return self.classifier(z)

# --- Optimization Wrappers ---

class QuantizableClassifier(nn.Module):
    """Wrapper to help with Eager Mode Quantization of sub-modules."""
    def __init__(self, classifier):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.classifier = classifier
        self.dequant = torch.ao.quantization.DeQuantStub()
    def forward(self, x):
        x = self.quant(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

class FP16Wrapper(nn.Module):
    """Wrapper to force FP16 inference on a model (useful for weight-only simulation)."""
    def __init__(self, model):
        super().__init__()
        self.model = model.half()
    def forward(self, x):
        return self.model(x.half()).float()

class ManualMixedPrecisionWrapper(nn.Module):
    """Wrapper to simulate mixed precision (FP16 weights, FP32 features)."""
    def __init__(self, model):
        super().__init__()
        self.features = model.features.float() # Ensure FP32
        self.classifier = model.classifier.half() # Ensure FP16
    def forward(self, x):
        z = self.features(x.float()).squeeze(-1)
        # Cast to half before classifier, back to float after
        out = self.classifier(z.half()).float()
        return out
