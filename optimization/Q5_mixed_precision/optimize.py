import os
import sys
from pathlib import Path
import torch
import torch.nn as nn

# Modular Imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.config import SEED, BASELINE_MODEL_PATH, OPTIMIZATION_DIR
from utils.eval_utils import evaluate_model
from models.ecg_net import ECGNet1D

class ManualMixedPrecisionWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = model.features.half()
        self.classifier = model.classifier.float()
    def forward(self, x):
        x = self.features(x.half())
        x = x.squeeze(-1)
        return self.classifier(x.float())

def main():
    torch.manual_seed(SEED)
    
    save_model_path = OPTIMIZATION_DIR / "Q5_model.pt"
    save_metrics_path = OPTIMIZATION_DIR / "Q5_metrics.json"
    os.makedirs(OPTIMIZATION_DIR, exist_ok=True)
    
    model = torch.load(BASELINE_MODEL_PATH, map_location="cpu", weights_only=False)
    model = ManualMixedPrecisionWrapper(model)
    
    print("Mixed Precision applied. Evaluating...")
    evaluate_model(
        model, 
        model_name="ECGNet1D_MixedPrecision", 
        technique_id="Q5", 
        technique_name="Mixed Precision (FP16+FP32)",
        save_path=save_metrics_path,
        device="cpu",
        save_model_path=save_model_path
    )

if __name__ == "__main__":
    main()
