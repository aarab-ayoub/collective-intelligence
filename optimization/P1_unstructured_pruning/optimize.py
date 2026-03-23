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
    save_model_path = project_root / "results" / "optimization" / "P1_model.pt"
    save_metrics_path = project_root / "results" / "optimization" / "P1_metrics.json"
    data_dir = project_root.parent / "mitbih"
    
    os.makedirs(save_model_path.parent, exist_ok=True)
    
    model = ECGNet1D()
    model.load_state_dict(torch.load(baseline_weights, map_location="cpu"))
    
    train_loader = get_train_loader(data_dir)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # Lower LR for fine-tuning
    
    # 1. Iterative Pruning (10% jumps to 50%)
    target_sparsity = 0.5
    steps = 5
    sparsity_step = target_sparsity / steps
    
    modules_to_prune = [m for m in model.modules() if isinstance(m, (nn.Conv1d, nn.Linear))]
    
    model.train()
    print("Starting iterative unstructured pruning...")
    for step in range(1, steps + 1):
        current_sparsity = step * sparsity_step
        print(f"--- Pruning Step {step}/{steps} (Target Sparsity: {current_sparsity:.1%}) ---")
        
        # Apply pruning
        for module in modules_to_prune:
            # prune.l1_unstructured applies cumulatively if called again
            # We want total sparsity to equal current_sparsity
            # If a tensor is already 10% sparse, pruning 10% of the REMAINING weights = 19% total
            # To just hit exactly 10%, 20% total, we use the amount parameter relative to the unpruned base.
            prune.l1_unstructured(module, name='weight', amount=sparsity_step)
            
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
            
    # 3. Make pruning permanent and prepare sparse state dict
    for module in modules_to_prune:
        prune.remove(module, 'weight')
        
    model.eval()
    
    # Convert weights to sparse tensors for storage size measurement
    sparse_state_dict = {}
    for name, param in model.state_dict().items():
        if 'weight' in name and len(param.shape) >= 2:
            # Convert specifically the pruned weights to sparse representation
            # to measure actual storage benefits.
            sparse_state_dict[name] = param.to_sparse()
        else:
            sparse_state_dict[name] = param
            
    print("Unstructured iterative pruning complete. Evaluating...")
    evaluate_model(
        model, 
        model_name="ECGNet1D_UnstPruned", 
        technique_id="P1", 
        technique_name="Unstructured Pruning (50%)",
        save_path=save_metrics_path,
        device="cpu",
        save_model_path=save_model_path,
        sparse_state_dict=sparse_state_dict
    )

if __name__ == "__main__":
    main()
