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
    save_model_path = project_root / "results" / "optimization" / "P3_model.pt"
    save_metrics_path = project_root / "results" / "optimization" / "P3_metrics.json"
    
    os.makedirs(save_model_path.parent, exist_ok=True)
    
    model = ECGNet1D()
    model.load_state_dict(torch.load(baseline_weights, map_location="cpu"))
    model.eval()
    
    # Apply 40% Global Magnitude pruning.
    # Instead of layer-by-layer, this prunes the lowest 40% weights ACROSS the entire network.
    
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv1d) or isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))
            
    # Global prune
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.4,
    )
    
    # Make permanent
    for module, name in parameters_to_prune:
        prune.remove(module, name)
            
    print("Global Magnitude Pruning (40%) applied. Evaluating...")
    evaluate_model(
        model, 
        model_name="ECGNet1D_GlobalMagPruned", 
        technique_id="P3", 
        technique_name="Global Magnitude Pruning (40%)",
        save_path=save_metrics_path,
        device="cpu",
        save_model_path=save_model_path
    )

if __name__ == "__main__":
    main()
