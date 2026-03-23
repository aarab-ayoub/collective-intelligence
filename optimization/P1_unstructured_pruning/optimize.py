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
    save_model_path = project_root / "results" / "optimization" / "P1_model.pt"
    save_metrics_path = project_root / "results" / "optimization" / "P1_metrics.json"
    
    os.makedirs(save_model_path.parent, exist_ok=True)
    
    model = ECGNet1D()
    model.load_state_dict(torch.load(baseline_weights, map_location="cpu"))
    model.eval()
    
    # Apply 50% unstructured L1 pruning to each Conv1d and Linear layer individually
    for module in model.modules():
        if isinstance(module, torch.nn.Conv1d) or isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.5)
            # Make pruning permanent
            prune.remove(module, 'weight')
            
    # Sparsity makes it compress better when saved to disk
    
    print("Unstructured pruning (50%) applied. Evaluating...")
    evaluate_model(
        model, 
        model_name="ECGNet1D_UnstPruned", 
        technique_id="P1", 
        technique_name="Unstructured Pruning (50%)",
        save_path=save_metrics_path,
        device="cpu",
        save_model_path=save_model_path
    )

if __name__ == "__main__":
    main()
