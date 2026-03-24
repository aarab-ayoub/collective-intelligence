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
    
    save_model_path = OPTIMIZATION_DIR / "P1_model.pt"
    save_metrics_path = OPTIMIZATION_DIR / "P1_metrics.json"
    os.makedirs(OPTIMIZATION_DIR, exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = torch.load(BASELINE_MODEL_PATH, map_location=device, weights_only=False)
    
    train_loader = get_train_loader()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    target_sparsity = 0.5
    steps = 5
    sparsity_step = target_sparsity / steps
    modules_to_prune = [m for m in model.modules() if isinstance(m, (nn.Conv1d, nn.Linear))]
    
    model.train()
    print("Starting iterative unstructured pruning...")
    for step in range(1, steps + 1):
        current_sparsity = step * sparsity_step
        print(f"--- Step {step}/{steps} (Sparsity: {current_sparsity:.1%}) ---")
        for module in modules_to_prune:
            prune.l1_unstructured(module, name='weight', amount=sparsity_step)
            
        for epoch in range(2):
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                
    for module in modules_to_prune:
        prune.remove(module, 'weight')
        
    model.eval()
    sparse_state_dict = {n: (p.to_sparse() if 'weight' in n and p.dim() >= 2 else p) for n, p in model.state_dict().items()}
    
    evaluate_model(
        model, 
        model_name="ECGNet1D_UnstPruned", 
        technique_id="P1", 
        technique_name="Unstructured Pruning (50%)",
        save_path=save_metrics_path,
        device=device,
        save_model_path=save_model_path,
        sparse_state_dict=sparse_state_dict
    )

if __name__ == "__main__":
    main()
