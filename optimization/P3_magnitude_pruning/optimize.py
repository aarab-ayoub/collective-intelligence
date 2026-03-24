import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# Modular Imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.config import SEED, BASELINE_MODEL_PATH, OPTIMIZATION_DIR
from utils.eval_utils import evaluate_model
from utils.data_loader import get_train_loader
from models.ecg_net import ECGNet1D

def main():
    torch.manual_seed(SEED)
    save_model_path = OPTIMIZATION_DIR / "P3_model.pt"
    save_metrics_path = OPTIMIZATION_DIR / "P3_metrics.json"
    os.makedirs(OPTIMIZATION_DIR, exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = torch.load(BASELINE_MODEL_PATH, map_location=device, weights_only=False)
    
    train_loader = get_train_loader()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    target_sparsity = 0.4
    steps = 4
    params_to_prune = [(m, 'weight') for m in model.modules() if isinstance(m, (nn.Conv1d, nn.Linear))]
    
    model.train()
    for step in range(1, steps + 1):
        s = step * (target_sparsity / steps)
        for m, name in params_to_prune: 
            if prune.is_pruned(m): prune.remove(m, name)
        prune.global_unstructured(params_to_prune, pruning_method=prune.L1Unstructured, amount=s)
        for epoch in range(2):
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad(); loss = criterion(model(bx), by); loss.backward(); optimizer.step()
                
    for m, name in params_to_prune: prune.remove(m, name)
    model.eval()
    sparse_state_dict = {n: (p.to_sparse() if 'weight' in n and p.dim() >= 2 else p) for n, p in model.state_dict().items()}
    evaluate_model(model, "ECGNet1D_GlobalMagPruned", "P3", "Global Magnitude Pruning (40%)", save_path=save_metrics_path, device=device, save_model_path=save_model_path, sparse_state_dict=sparse_state_dict)

if __name__ == "__main__":
    main()
