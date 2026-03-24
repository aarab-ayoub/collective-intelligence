import os
import sys
from pathlib import Path
import torch
import torch.nn as nn

# Modular Imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.config import SEED, BASELINE_MODEL_PATH, OPTIMIZATION_DIR
from utils.eval_utils import evaluate_model
from utils.data_loader import get_train_loader
from models.ecg_net import ECGNet1D

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

def get_topk_indices(weight_tensor, keep_ratio=0.7):
    l1_norms = weight_tensor.abs().sum(dim=list(range(1, weight_tensor.dim())))
    num_keep = max(1, int(weight_tensor.size(0) * keep_ratio))
    _, indices = torch.topk(l1_norms, num_keep)
    return indices.sort()[0]

def main():
    torch.manual_seed(SEED)
    save_model_path = OPTIMIZATION_DIR / "P2_model.pt"
    save_metrics_path = OPTIMIZATION_DIR / "P2_metrics.json"
    os.makedirs(OPTIMIZATION_DIR, exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = torch.load(BASELINE_MODEL_PATH, map_location=device, weights_only=False)
    
    keep_ratio = 0.7
    conv_indices = [0, 3, 8, 11, 16]
    kept_indices_map = {idx: get_topk_indices(model.features[idx].weight.data, keep_ratio) for idx in conv_indices}
    kept_fc1_idx = get_topk_indices(model.classifier[1].weight.data, keep_ratio)
    
    narrow_model = ECGNet1D_Narrow(
        c1=len(kept_indices_map[0]), c2=len(kept_indices_map[3]),
        c3=len(kept_indices_map[8]), c4=len(kept_indices_map[11]),
        c5=len(kept_indices_map[16]), fc1=len(kept_fc1_idx)
    ).to(device)
    
    print("Copying weights to narrow model...")
    # Conv 0
    idx = 0
    curr_kept = kept_indices_map[idx]
    narrow_model.features[idx].weight.data.copy_(model.features[idx].weight.data[curr_kept])
    narrow_model.features[idx].bias.data.copy_(model.features[idx].bias.data[curr_kept])
    # BN 1
    narrow_model.features[idx+1].weight.data.copy_(model.features[idx+1].weight.data[curr_kept])
    narrow_model.features[idx+1].bias.data.copy_(model.features[idx+1].bias.data[curr_kept])
    narrow_model.features[idx+1].running_mean.copy_(model.features[idx+1].running_mean[curr_kept])
    narrow_model.features[idx+1].running_var.copy_(model.features[idx+1].running_var[curr_kept])
    prev_kept = curr_kept

    # Conv 3
    idx = 3
    curr_kept = kept_indices_map[idx]
    narrow_model.features[idx].weight.data.copy_(model.features[idx].weight.data[curr_kept][:, prev_kept])
    narrow_model.features[idx].bias.data.copy_(model.features[idx].bias.data[curr_kept])
    # BN 4
    narrow_model.features[idx+1].weight.data.copy_(model.features[idx+1].weight.data[curr_kept])
    narrow_model.features[idx+1].bias.data.copy_(model.features[idx+1].bias.data[curr_kept])
    narrow_model.features[idx+1].running_mean.copy_(model.features[idx+1].running_mean[curr_kept])
    narrow_model.features[idx+1].running_var.copy_(model.features[idx+1].running_var[curr_kept])
    prev_kept = curr_kept

    # Conv 8
    idx = 8
    curr_kept = kept_indices_map[idx]
    narrow_model.features[idx].weight.data.copy_(model.features[idx].weight.data[curr_kept][:, prev_kept])
    narrow_model.features[idx].bias.data.copy_(model.features[idx].bias.data[curr_kept])
    # BN 9
    narrow_model.features[idx+1].weight.data.copy_(model.features[idx+1].weight.data[curr_kept])
    narrow_model.features[idx+1].bias.data.copy_(model.features[idx+1].bias.data[curr_kept])
    narrow_model.features[idx+1].running_mean.copy_(model.features[idx+1].running_mean[curr_kept])
    narrow_model.features[idx+1].running_var.copy_(model.features[idx+1].running_var[curr_kept])
    prev_kept = curr_kept

    # Conv 11
    idx = 11
    curr_kept = kept_indices_map[idx]
    narrow_model.features[idx].weight.data.copy_(model.features[idx].weight.data[curr_kept][:, prev_kept])
    narrow_model.features[idx].bias.data.copy_(model.features[idx].bias.data[curr_kept])
    # BN 12
    narrow_model.features[idx+1].weight.data.copy_(model.features[idx+1].weight.data[curr_kept])
    narrow_model.features[idx+1].bias.data.copy_(model.features[idx+1].bias.data[curr_kept])
    narrow_model.features[idx+1].running_mean.copy_(model.features[idx+1].running_mean[curr_kept])
    narrow_model.features[idx+1].running_var.copy_(model.features[idx+1].running_var[curr_kept])
    prev_kept = curr_kept

    # Conv 16
    idx = 16
    curr_kept = kept_indices_map[idx]
    narrow_model.features[idx].weight.data.copy_(model.features[idx].weight.data[curr_kept][:, prev_kept])
    narrow_model.features[idx].bias.data.copy_(model.features[idx].bias.data[curr_kept])
    # BN 17
    narrow_model.features[idx+1].weight.data.copy_(model.features[idx+1].weight.data[curr_kept])
    narrow_model.features[idx+1].bias.data.copy_(model.features[idx+1].bias.data[curr_kept])
    narrow_model.features[idx+1].running_mean.copy_(model.features[idx+1].running_mean[curr_kept])
    narrow_model.features[idx+1].running_var.copy_(model.features[idx+1].running_var[curr_kept])
    prev_kept = curr_kept

    # Classifier 1
    narrow_model.classifier[1].weight.data.copy_(model.classifier[1].weight.data[kept_fc1_idx][:, prev_kept])
    narrow_model.classifier[1].bias.data.copy_(model.classifier[1].bias.data[kept_fc1_idx])
    # Classifier 4
    narrow_model.classifier[4].weight.data.copy_(model.classifier[4].weight.data[:, kept_fc1_idx])
    narrow_model.classifier[4].bias.data.copy_(model.classifier[4].bias.data)

    print("Fine-tuning structured-pruned model...")
    train_loader = get_train_loader()
    optimizer = torch.optim.Adam(narrow_model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    narrow_model.train()
    for epoch in range(5):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(narrow_model(bx), by)
            loss.backward()
            optimizer.step()

    evaluate_model(
        narrow_model, 
        model_name="ECGNet1D_Narrow_StructPruned", 
        technique_id="P2", 
        technique_name="Structured Pruning (30%)",
        save_path=save_metrics_path,
        device=device,
        save_model_path=save_model_path
    )

if __name__ == "__main__":
    main()
