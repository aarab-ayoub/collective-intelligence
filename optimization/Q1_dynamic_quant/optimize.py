import os
import sys
from pathlib import Path
import torch
import torch.quantization

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import ECGNet1D, evaluate_model, SEED

def main():
    torch.manual_seed(SEED)
    
    # Set quantization engine for Mac/ARM compatibility
    torch.backends.quantized.engine = 'qnnpack'
    
    # Paths
    project_root = Path(__file__).resolve().parent.parent.parent
    baseline_weights = project_root / "baseline" / "weights" / "baseline_mitbih_csv.pt"
    save_model_path = project_root / "results" / "optimization" / "Q1_model.pt"
    save_metrics_path = project_root / "results" / "optimization" / "Q1_metrics.json"
    
    os.makedirs(save_model_path.parent, exist_ok=True)
    
    # Load baseline
    model = ECGNet1D()
    model.load_state_dict(torch.load(baseline_weights, map_location="cpu"))
    model.eval()
    
    # 1. Dynamic Quantization
    # Only applies to nn.Linear by default
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )
    
    print("Baseline dynamic quantization applied. Evaluating...")
    
    evaluate_model(
        quantized_model, 
        model_name="ECGNet1D_DynQuant", 
        technique_id="Q1", 
        technique_name="Dynamic Quantization",
        save_path=save_metrics_path,
        device="cpu",  # Quantization usually evaluated on CPU
        save_model_path=save_model_path
    )

if __name__ == "__main__":
    main()
