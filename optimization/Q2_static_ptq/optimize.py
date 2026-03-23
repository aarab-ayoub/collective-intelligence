import os
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import ECGNet1D, evaluate_model, SEED

def main():
    torch.manual_seed(SEED)
    torch.backends.quantized.engine = 'qnnpack'
    
    project_root = Path(__file__).resolve().parent.parent.parent
    baseline_weights = project_root / "baseline" / "weights" / "baseline_mitbih_csv.pt"
    save_model_path = project_root / "results" / "optimization" / "Q2_model.pt"
    save_metrics_path = project_root / "results" / "optimization" / "Q2_metrics.json"
    data_dir = project_root.parent / "mitbih"
    
    os.makedirs(save_model_path.parent, exist_ok=True)
    
    # Load baseline
    model = ECGNet1D()
    model.load_state_dict(torch.load(baseline_weights, map_location="cpu"))
    model.eval()
    
    # To avoid Mac ARM Conv1d quantization bugs, we'll static-PTQ the classifier only.
    # The feature extractor remains unquantized.
    
    # Wrap classifier with stubs
    class QuantizableClassifier(torch.nn.Module):
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

    # Replace classifier with quantizable version
    model.classifier = QuantizableClassifier(model.classifier)
    
    # Set qconfig for the classifier ONLY
    model.classifier.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    
    # Prepare
    torch.quantization.prepare(model.classifier, inplace=True)
    
    # Calibrate
    print("Calibrating the classifier...")
    train_df = pd.read_csv(data_dir / "mitbih_train.csv", header=None, nrows=1024)
    X_calib = train_df.iloc[:, :-1].values.astype(np.float32)
    x_calib_t = torch.tensor(X_calib).unsqueeze(1)
    calib_loader = DataLoader(TensorDataset(x_calib_t), batch_size=128, shuffle=False)
    
    with torch.no_grad():
        for bx, in calib_loader:
            model(bx)
            
    # Convert
    print("Converting...")
    torch.quantization.convert(model.classifier, inplace=True)
    
    print("Static PTQ (Classifier-only) applied. Evaluating...")
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
