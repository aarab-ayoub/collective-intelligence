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

class FP16Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.half()
    def forward(self, x):
        return self.model(x.half()).float()

def main():
    torch.manual_seed(SEED)
    
    save_model_path = OPTIMIZATION_DIR / "Q4_model.pt"
    save_metrics_path = OPTIMIZATION_DIR / "Q4_metrics.json"
    os.makedirs(OPTIMIZATION_DIR, exist_ok=True)
    
    model = torch.load(BASELINE_MODEL_PATH, map_location="cpu", weights_only=False)
    model = FP16Wrapper(model)
    
    print("Weight-only (FP16) applied. Evaluating...")
    evaluate_model(
        model, 
        model_name="ECGNet1D_WeightOnly", 
        technique_id="Q4", 
        technique_name="Weight-only Quantization (FP16)",
        save_path=save_metrics_path,
        device="cpu",
        save_model_path=save_model_path
    )

if __name__ == "__main__":
    main()
