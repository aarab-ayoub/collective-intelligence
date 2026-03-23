import os
import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.utils import ECGNet1D, evaluate_model, SEED

class ManualMixedPrecisionWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Manually cast features (conv layers) to FP16
        self.model.features = self.model.features.half()
        # Ensure classifier remains in FP32
        self.model.classifier = self.model.classifier.float()

    def forward(self, x):
        # Cast input to FP16 for the features part
        x_fp16 = x.to(torch.float16)
        z = self.model.features(x_fp16)
        # Cast back to FP32 for the classifier part
        z_fp32 = z.to(torch.float32)
        # Need to squeeze for the AdaptiveAvgPool output shape
        return self.model.classifier(z_fp32.squeeze(-1))

def main():
    torch.manual_seed(SEED)
    
    project_root = Path(__file__).resolve().parent.parent.parent
    baseline_model_path = project_root / "results" / "baseline" / "baseline_best.pt"
    save_model_path = project_root / "results" / "optimization" / "Q5_model.pt"
    save_metrics_path = project_root / "results" / "optimization" / "Q5_metrics.json"
    
    os.makedirs(save_model_path.parent, exist_ok=True)
    
    model = torch.load(baseline_model_path, map_location="cpu", weights_only=False)
    model.eval()
    
    wrapped_model = ManualMixedPrecisionWrapper(model)
    wrapped_model.eval()
    
    print("Mixed precision (FP16 Features + FP32 Classifier) applied. Evaluating...")
    
    evaluate_model(
        wrapped_model, 
        model_name="ECGNet1D_MixedPrec_Manual", 
        technique_id="Q5", 
        technique_name="Mixed Precision (FP16+FP32)",
        save_path=save_metrics_path,
        device="cpu", # Fallback to prevent native MPS errors with half types
        save_model_path=save_model_path
    )

if __name__ == "__main__":
    main()
