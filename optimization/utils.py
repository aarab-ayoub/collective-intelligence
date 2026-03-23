import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Set the same random seed for reproducibility
SEED = 42

def load_test_data(data_dir="/Users/ayoub/work/MS-DS_ML_Projects/IOT/mitbih"):
    """Loads only the test data from the MIT-BIH CSV dataset."""
    test_df = pd.read_csv(Path(data_dir) / "mitbih_test.csv", header=None)
    X = test_df.iloc[:, :-1].values.astype(np.float32)
    y = test_df.iloc[:, -1].values.astype(np.int64)
    return X, y

def get_test_loader(batch_size=256):
    X_test, y_test = load_test_data()
    x_test_t = torch.tensor(X_test).unsqueeze(1)
    y_test_t = torch.tensor(y_test)
    test_loader = DataLoader(TensorDataset(x_test_t, y_test_t), batch_size=batch_size, shuffle=False)
    return test_loader

# Define the model architecture exactly as in Phase 1
class ECGNet1D(nn.Module):
    def __init__(self, n_classes=5, input_len=186, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.15),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
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

def evaluate_model(model, model_name, technique_id, technique_name, save_path=None, test_loader=None, device="cpu", save_model_path=None):
    """
    Evaluates a model (accuracy, F1, size, inference time) and saves metrics to a JSON file.
    Note: For optimized deployment timing, we evaluate strictly on CPU by default.
    """
    if test_loader is None:
        test_loader = get_test_loader()
        
    model.eval()
    model.to(device)
    
    test_preds, test_targs = [], []
    
    # Warmup
    with torch.no_grad():
        for bx, _ in test_loader:
            bx = bx.to(device)
            _ = model(bx)
            break
            
    # Measure inference time over the whole test set
    t0 = time.perf_counter()
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            logits = model(bx)
            test_preds.extend(logits.argmax(1).cpu().numpy())
            test_targs.extend(by.numpy())
            
    inference_time = (time.perf_counter() - t0) * 1000  # ms
    avg_inference_ms = inference_time / len(test_targs)
    
    test_preds = np.array(test_preds)
    test_targs = np.array(test_targs)
    
    acc = accuracy_score(test_targs, test_preds)
    f1 = f1_score(test_targs, test_preds, average="macro", zero_division=0)
    
    # Save model to disk temporarily if path provided to measure exact size
    if save_model_path:
        torch.save(model.state_dict() if hasattr(model, 'state_dict') else model, save_model_path)
        model_size_bytes = os.path.getsize(save_model_path)
    else:
        # Fallback approximation: serialize to memory
        import io
        buffer = io.BytesIO()
        torch.save(model.state_dict() if hasattr(model, 'state_dict') else model, buffer)
        model_size_bytes = buffer.getbuffer().nbytes
        
    model_size_mb = model_size_bytes / (1024 * 1024)
    
    metrics = {
        "id": technique_id,
        "technique": technique_name,
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "size_mb": float(model_size_mb),
        "inference_ms": float(avg_inference_ms)
    }
    
    print(f"--- {technique_name} ({technique_id}) ---")
    print(f"Accuracy    : {acc*100:.2f}%")
    print(f"Macro F1    : {f1:.4f}")
    print(f"Size        : {model_size_mb:.3f} MB")
    print(f"Inv. Time   : {avg_inference_ms:.4f} ms/sample")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=2)
            
    return metrics
