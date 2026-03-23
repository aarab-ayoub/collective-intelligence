import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.utils import ECGNet1D, evaluate_model, SEED

def main():
    torch.manual_seed(SEED)
    torch.backends.quantized.engine = 'qnnpack'
    
    project_root = Path(__file__).resolve().parent.parent.parent
    baseline_model_path = project_root / "results" / "baseline" / "baseline_best.pt"
    save_model_path = project_root / "results" / "optimization" / "Q3_model.pt"
    save_metrics_path = project_root / "results" / "optimization" / "Q3_metrics.json"
    data_dir = project_root.parent / "mitbih"
    
    os.makedirs(save_model_path.parent, exist_ok=True)
    
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

def main():
    torch.manual_seed(SEED)
    torch.backends.quantized.engine = 'qnnpack'
    
    project_root = Path(__file__).resolve().parent.parent.parent
    baseline_model_path = project_root / "results" / "baseline" / "baseline_best.pt"
    save_model_path = project_root / "results" / "optimization" / "Q3_model.pt"
    save_metrics_path = project_root / "results" / "optimization" / "Q3_metrics.json"
    data_dir = project_root.parent / "mitbih"
    
    os.makedirs(save_model_path.parent, exist_ok=True)
    
    model = torch.load(baseline_model_path, map_location="cpu", weights_only=False)
    
    # Like Q2, wrap only the classifier to avoid Conv1d Mac bugs
    model.classifier = QuantizableClassifier(model.classifier)
    
    # QAT config
    model.classifier.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
    
    # Prepare QAT
    torch.quantization.prepare_qat(model.classifier, inplace=True)
    
    # Fine-tune with QAT briefly (1 epoch, 10% data)
    print("Fine-tuning QAT...")
    model.train()
    
    train_df = pd.read_csv(data_dir / "mitbih_train.csv", header=None)
    # Use 10% of train data for quick fine-tuning
    subset_df = train_df.sample(frac=0.1, random_state=SEED)
    X_train = subset_df.iloc[:, :-1].values.astype(np.float32)
    y_train = subset_df.iloc[:, -1].values.astype(np.int64)
    
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train).unsqueeze(1), torch.tensor(y_train)), 
        batch_size=128, shuffle=True
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1):
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
    
    model.eval()
    print("Converting QAT model to quantized model...")
    torch.quantization.convert(model.classifier, inplace=True)
    
    print("QAT applied. Evaluating...")
    evaluate_model(
        model, 
        model_name="ECGNet1D_QAT", 
        technique_id="Q3", 
        technique_name="Quantization-Aware Training",
        save_path=save_metrics_path,
        device="cpu",
        save_model_path=save_model_path
    )

if __name__ == "__main__":
    main()
