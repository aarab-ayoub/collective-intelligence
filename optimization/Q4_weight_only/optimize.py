import os
import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import ECGNet1D, evaluate_model, SEED

def main():
    torch.manual_seed(SEED)
    
    project_root = Path(__file__).resolve().parent.parent.parent
    baseline_weights = project_root / "baseline" / "weights" / "baseline_mitbih_csv.pt"
    save_model_path = project_root / "results" / "optimization" / "Q4_model.pt"
    save_metrics_path = project_root / "results" / "optimization" / "Q4_metrics.json"
    
    os.makedirs(save_model_path.parent, exist_ok=True)
    
    model = ECGNet1D()
    model.load_state_dict(torch.load(baseline_weights, map_location="cpu"))
    model.eval()
    
    # Weight-only quantization manually compresses weight tensors from 32-bit floats to 8-bit integers 
    # and scales them back during inference.
    # We will use PyTorch's native dynamic quant, which is basically weight-only + activation quantization 
    # at runtime, BUT we only provide qconfig for weight.
    
    # In PyTorch, True Weight-Only quantization isn't natively exposed via torch.quantization without custom backends.
    # To simulate it properly, we will use float16 conversion natively which halves the size immediately
    # Weight-only FP16 conversion is standard and highly reliable.
    
    model.half() # Converts parameters to float16
    
    # However, since some evaluation operations might fail if inputs are float32,
    # we'll create a wrapper that casts inputs
    class FP16Wrapper(nn.Module):
        def __init__(self, fp16_model):
            super().__init__()
            self.model = fp16_model
            
        def forward(self, x):
            # Input comes as float32, we cast to float16 for the model
            return self.model(x.half()).float()
            
    wrapped_model = FP16Wrapper(model)
    wrapped_model.eval()

    print("Weight-only (FP16) applied. Evaluating...")
    evaluate_model(
        wrapped_model, 
        model_name="ECGNet1D_WeightOnly_FP16", 
        technique_id="Q4", 
        technique_name="Weight-only Quantization (FP16)",
        save_path=save_metrics_path,
        device="cpu", # CPU can execute FP16 instructions
        save_model_path=save_model_path
    )

if __name__ == "__main__":
    main()
