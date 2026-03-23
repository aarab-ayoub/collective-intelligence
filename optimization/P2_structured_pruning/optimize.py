import os
import sys
from pathlib import Path
import torch
import torch.nn.utils.prune as prune

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import ECGNet1D, evaluate_model, SEED

def main():
    torch.manual_seed(SEED)
    
    project_root = Path(__file__).resolve().parent.parent.parent
    baseline_weights = project_root / "baseline" / "weights" / "baseline_mitbih_csv.pt"
    save_model_path = project_root / "results" / "optimization" / "P2_model.pt"
    save_metrics_path = project_root / "results" / "optimization" / "P2_metrics.json"
    
    os.makedirs(save_model_path.parent, exist_ok=True)
    
    model = ECGNet1D()
    model.load_state_dict(torch.load(baseline_weights, map_location="cpu"))
    model.eval()
    
    # Apply 30% structured pruning: drop entire filters (channels) based on L1 norm
    # Structured pruning physically removes compute paths (unlike unstructured which just zeroes weights)
    
    for module in model.modules():
        if isinstance(module, torch.nn.Conv1d):
            # Prune 30% of output channels (dim=0)
            prune.ln_structured(module, name='weight', amount=0.3, n=1, dim=0)
            prune.remove(module, 'weight')
        elif isinstance(module, torch.nn.Linear):
            # For linear layers, prune 30% of output rows
            # except the final classifier layer which must stay 5 output classes
            if module.out_features != 5:
                prune.ln_structured(module, name='weight', amount=0.3, n=1, dim=0)
                prune.remove(module, 'weight')
            
    print("Structured pruning (30% filters/rows) applied. Evaluating...")
    evaluate_model(
        model, 
        model_name="ECGNet1D_StructPruned", 
        technique_id="P2", 
        technique_name="Structured Pruning (30%)",
        save_path=save_metrics_path,
        device="cpu",
        save_model_path=save_model_path
    )

if __name__ == "__main__":
    main()
