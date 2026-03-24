import os
import sys
from pathlib import Path
import torch
import torch.quantization

# Add parent directory to path to import models and utils
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.config import SEED, BASELINE_MODEL_PATH, OPTIMIZATION_DIR
from utils.eval_utils import evaluate_model
from models.ecg_net import ECGNet1D

def main():
    torch.manual_seed(SEED)
    torch.backends.quantized.engine = 'qnnpack'
    
    save_model_path = OPTIMIZATION_DIR / "Q1_model.pt"
    save_metrics_path = OPTIMIZATION_DIR / "Q1_metrics.json"
    
    os.makedirs(save_model_path.parent, exist_ok=True)
    
    # Load baseline
    model = torch.load(BASELINE_MODEL_PATH, map_location="cpu", weights_only=False)
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
