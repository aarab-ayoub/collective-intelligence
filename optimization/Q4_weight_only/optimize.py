import os
import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.utils import ECGNet1D, evaluate_model, SEED

class FP16Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Convert all Conv1d and Linear weights to float16
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)):
                module.weight.data = module.weight.data.to(torch.float16)
                if getattr(module, 'bias', None) is not None:
                    module.bias.data = module.bias.data.to(torch.float16)

    def forward(self, x):
        return self.model(x.half()).float()

def main():
    torch.manual_seed(SEED)
    
    project_root = Path(__file__).resolve().parent.parent.parent
    baseline_model_path = project_root / "results" / "baseline" / "baseline_best.pt"
    save_model_path = project_root / "results" / "optimization" / "Q4_model.pt"
    save_metrics_path = project_root / "results" / "optimization" / "Q4_metrics.json"
    
    os.makedirs(save_model_path.parent, exist_ok=True)
    
    model = torch.load(baseline_model_path, map_location="cpu", weights_only=False)
    model.eval()
    
    model.half()
            
    wrapped_model = FP16Wrapper(model)
    wrapped_model.eval()
    
    print("Weight-only (FP16) applied. Evaluating...")
    
    evaluate_model(
        wrapped_model, 
        model_name="ECGNet1D_WeightOnly_FP16", 
        technique_id="Q4", 
        technique_name="Weight-only Quantization (FP16)",
        save_path=save_metrics_path,
        device="cpu", # Fallback calculation on cpu to prevent native MPS errors with half types
        save_model_path=save_model_path
    )

if __name__ == "__main__":
    main()
