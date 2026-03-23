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
    save_model_path = project_root / "results" / "optimization" / "Q5_model.pt"
    save_metrics_path = project_root / "results" / "optimization" / "Q5_metrics.json"
    
    os.makedirs(save_model_path.parent, exist_ok=True)
    
    model = ECGNet1D()
    model.load_state_dict(torch.load(baseline_weights, map_location="cpu"))
    model.eval()
    
    # Custom Mixed Precision Wrapper
    # To avoid PyTorch Mac CPU autocast deadlocks with bfloat16,
    # we manually orchestrate mixed precision without context managers.
    # We run the Conv1D feature extractor in FP16 (which saves memory & compute)
    # and the Linear classifier in FP32 (which is sensitive to numerical instability).
    
    model.features.half()  # Convert feature extractor to FP16
    
    class ManualMixedPrecisionWrapper(nn.Module):
        def __init__(self, mixed_model):
            super().__init__()
            self.model = mixed_model
            
        def forward(self, x):
            # Input to FP16 for the features block
            x_fp16 = self.model.features(x.half())
            # Convert back to FP32 for the classifier block
            x_fp32 = x_fp16.float().squeeze(-1)
            return self.model.classifier(x_fp32)
                
    wrapped_model = ManualMixedPrecisionWrapper(model)
    wrapped_model.eval()

    print("Mixed precision (FP16 Features + FP32 Classifier) applied. Evaluating...")
    evaluate_model(
        wrapped_model, 
        model_name="ECGNet1D_MixedPrecision", 
        technique_id="Q5", 
        technique_name="Mixed Precision (FP16+FP32)",
        save_path=save_metrics_path,
        device="cpu",
        save_model_path=save_model_path
    )

if __name__ == "__main__":
    main()
