import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import ECGNet1D, evaluate_model, SEED

def get_train_loader(data_dir, batch_size=256):
    train_df = pd.read_csv(Path(data_dir) / "mitbih_train.csv", header=None)
    X = train_df.iloc[:, :-1].values.astype(np.float32)
    y = train_df.iloc[:, -1].values.astype(np.int64)
    return DataLoader(TensorDataset(torch.tensor(X).unsqueeze(1), torch.tensor(y)), batch_size=batch_size, shuffle=True)

def main():
    torch.manual_seed(SEED)
    
    project_root = Path(__file__).resolve().parent.parent.parent
    baseline_weights = project_root / "baseline" / "weights" / "baseline_mitbih_csv.pt"
    save_model_path = project_root / "results" / "optimization" / "P3_model.pt"
    save_metrics_path = project_root / "results" / "optimization" / "P3_metrics.json"
    data_dir = project_root.parent / "mitbih"
    
    os.makedirs(save_model_path.parent, exist_ok=True)
    
    model = ECGNet1D()
    model.load_state_dict(torch.load(baseline_weights, map_location="cpu"))
    
    train_loader = get_train_loader(data_dir)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 1. Iterative Global Magnitude Pruning (10% jumps to 40%)
    target_sparsity = 0.4
    steps = 4
    sparsity_step = target_sparsity / steps
    
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
            
    model.train()
    print("Starting iterative global magnitude pruning...")
    for step in range(1, steps + 1):
        target_step_sparsity = step * sparsity_step
        print(f"--- Pruning Step {step}/{steps} (Target Sparsity: {target_step_sparsity:.1%}) ---")
        
        # When applying global_unstructured iteratively, PyTorch calculates the amount 
        # relative to the active (unpruned) weights. To reach target_step_sparsity overall,
        # we configure amount=sparsity_step. E.g., removing 10% of remaining weights each time
        # yields ~34% overall after 4 steps. To properly hit 40%, we just ask for `target_step_sparsity`
        # on the ORIGINAL dense model. However, PyTorch's pruning applies masks on top of masks if chained.
        # Let's remove old masks first to apply the new exact global threshold.
        for module, name in parameters_to_prune:
            if prune.is_pruned(module):
                prune.remove(module, name)
                
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=target_step_sparsity,
        )
        
        # Fine-tune for 2 epochs
        for epoch in range(2):
            epoch_loss = 0.0
            for bx, by in train_loader:
                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"  Fine-tune Epoch {epoch+1}/2 - Loss: {epoch_loss/len(train_loader):.4f}")
            
    # Make pruning permanent
    for module, name in parameters_to_prune:
        prune.remove(module, name)
        
    model.eval()
    
    # Sparse format storage calculation
    sparse_state_dict = {}
    for name, param in model.state_dict().items():
        if 'weight' in name and len(param.shape) >= 2:
            sparse_state_dict[name] = param.to_sparse()
        else:
            sparse_state_dict[name] = param
            
    print("Global iterative pruning complete. Evaluating...")
    evaluate_model(
        model, 
        model_name="ECGNet1D_GlobalMagPruned", 
        technique_id="P3", 
        technique_name="Global Magnitude Pruning (40%)",
        save_path=save_metrics_path,
        device="cpu",
        save_model_path=save_model_path,
        sparse_state_dict=sparse_state_dict
    )

if __name__ == "__main__":
    main()
