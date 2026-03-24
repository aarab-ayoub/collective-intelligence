import os
import sys
from pathlib import Path
import torch
import torch.nn as nn

# Modular Imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.config import SEED, BASELINE_MODEL_PATH, OPTIMIZATION_DIR, DATA_DIR
from utils.eval_utils import evaluate_model
from models.ecg_net import ECGNet1D

class QuantizableClassifier(nn.Module):
    def __init__(self, classifier_module):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.classifier = classifier_module
        self.dequant = torch.quantization.DeQuantStub()
        
    def forward(self, x):
        x = self.quant(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

def main():
    torch.manual_seed(SEED)
    torch.backends.quantized.engine = 'qnnpack'
    
    save_model_path = OPTIMIZATION_DIR / "Q2_model.pt"
    save_metrics_path = OPTIMIZATION_DIR / "Q2_metrics.json"
    os.makedirs(OPTIMIZATION_DIR, exist_ok=True)
    
    # Load baseline
    model = torch.load(BASELINE_MODEL_PATH, map_location="cpu", weights_only=False)
    model.eval()
    
    # Wrap classifier
    model.classifier = QuantizableClassifier(model.classifier)
    model.classifier.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    
    # Prepare
    torch.quantization.prepare(model.classifier, inplace=True)
    
    # Calibrate
    print("Calibrating...")
    from torch.utils.data import DataLoader, TensorDataset
    import pandas as pd
    import numpy as np
    
    train_df = pd.read_csv(DATA_DIR / "mitbih_train.csv", header=None, nrows=1024)
    X_calib = train_df.iloc[:, :-1].values.astype(np.float32)
    x_calib_t = torch.tensor(X_calib).unsqueeze(1)
    calib_loader = DataLoader(TensorDataset(x_calib_t), batch_size=128, shuffle=False)
    
    with torch.no_grad():
        for bx, in calib_loader:
            model(bx)
            
    # Convert
    print("Converting...")
    torch.quantization.convert(model.classifier, inplace=True)
    
    print("Static PTQ applied. Evaluating...")
    evaluate_model(
        model, 
        model_name="ECGNet1D_StaticPTQ", 
        technique_id="Q2", 
        technique_name="Static PTQ",
        save_path=save_metrics_path,
        device="cpu",
        save_model_path=save_model_path
    )

if __name__ == "__main__":
    main()
