import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

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
    
    save_model_path = OPTIMIZATION_DIR / "Q3_model.pt"
    save_metrics_path = OPTIMIZATION_DIR / "Q3_metrics.json"
    os.makedirs(OPTIMIZATION_DIR, exist_ok=True)
    
    model = torch.load(BASELINE_MODEL_PATH, map_location="cpu", weights_only=False)
    model.classifier = QuantizableClassifier(model.classifier)
    model.classifier.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
    
    torch.quantization.prepare_qat(model.classifier, inplace=True)
    
    print("Fine-tuning QAT...")
    model.train()
    
    train_df = pd.read_csv(DATA_DIR / "mitbih_train.csv", header=None)
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
    print("Converting...")
    torch.quantization.convert(model.classifier, inplace=True)
    
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
