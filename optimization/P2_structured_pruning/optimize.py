import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import evaluate_model, SEED

def get_train_loader(data_dir, batch_size=256):
    train_df = pd.read_csv(Path(data_dir) / "mitbih_train.csv", header=None)
    X = train_df.iloc[:, :-1].values.astype(np.float32)
    y = train_df.iloc[:, -1].values.astype(np.int64)
    return DataLoader(TensorDataset(torch.tensor(X).unsqueeze(1), torch.tensor(y)), batch_size=batch_size, shuffle=True)

class ECGNet1D_Narrow(nn.Module):
    """
    A dynamically sizable version of ECGNet1D that accepts channel counts.
    """
    def __init__(self, c1, c2, c3, c4, c5, fc1, n_classes=5, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=7, padding=3),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.15),
            
            nn.Conv1d(c2, c3, kernel_size=5, padding=2),
            nn.BatchNorm1d(c3),
            nn.ReLU(inplace=True),
            nn.Conv1d(c3, c4, kernel_size=3, padding=1),
            nn.BatchNorm1d(c4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.15),
            
            nn.Conv1d(c4, c5, kernel_size=3, padding=1),
            nn.BatchNorm1d(c5),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(c5, fc1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(fc1, n_classes),
        )

    def forward(self, x):
        z = self.features(x).squeeze(-1)
        return self.classifier(z)

def get_topk_indices(weight_tensor, keep_ratio=0.7):
    # L1 norm across all dimensions except output channels (dim 0)
    l1_norms = weight_tensor.abs().sum(dim=list(range(1, weight_tensor.dim())))
    num_keep = int(weight_tensor.size(0) * keep_ratio)
    # Get indices of the top-k norms
    _, indices = torch.topk(l1_norms, num_keep)
    return indices.sort()[0] # Sort to maintain channel order

def main():
    torch.manual_seed(SEED)
    
    project_root = Path(__file__).resolve().parent.parent.parent
    baseline_weights = project_root / "baseline" / "weights" / "baseline_mitbih_csv.pt"
    save_model_path = project_root / "results" / "optimization" / "P2_model.pt"
    save_metrics_path = project_root / "results" / "optimization" / "P2_metrics.json"
    data_dir = project_root.parent / "mitbih"
    
    os.makedirs(save_model_path.parent, exist_ok=True)
    
    # Needs to import the dense model from utils to avoid circular dependency
    from utils import ECGNet1D
    model = ECGNet1D()
    model.load_state_dict(torch.load(baseline_weights, map_location="cpu"))
    
    # 1. Determine importance of filters/channels and keep 70%
    keep_ratio = 0.7
    
    # Feature layer indices with Conv1d
    conv_indices = [0, 3, 8, 11, 16]
    bn_indices = [1, 4, 9, 12, 17]
    
    # Classifier linear indices
    linear1_idx = 1
    # Final layer doesn't get pruned on output dimension
    linear2_idx = 4
    
    kept_out_channels = []
    
    # Map from layer idx to kept output indices
    kept_indices_map_feat = {}
    
    for idx in conv_indices:
        weight = model.features[idx].weight.data
        idxs = get_topk_indices(weight, keep_ratio)
        kept_indices_map_feat[idx] = idxs
        kept_out_channels.append(len(idxs))
        
    # Linear 1
    w_fc1 = model.classifier[linear1_idx].weight.data
    kept_fc1_idx = get_topk_indices(w_fc1, keep_ratio)
    fc1_out_ch = len(kept_fc1_idx)
    
    # 2. Architect the new dense narrow model
    narrow_model = ECGNet1D_Narrow(
        c1=kept_out_channels[0],
        c2=kept_out_channels[1],
        c3=kept_out_channels[2],
        c4=kept_out_channels[3],
        c5=kept_out_channels[4],
        fc1=fc1_out_ch
    )
    
    # 3. Copy kept weights into the narrow model
    
    # Feature block
    prev_kept_idx = None
    for i in range(len(model.features)):
        if isinstance(model.features[i], nn.Conv1d):
            orig_w = model.features[i].weight.data
            orig_b = model.features[i].bias.data
            
            curr_kept_idx = kept_indices_map_feat[i]
            
            # Slice output channels
            sliced_w = orig_w[curr_kept_idx]
            sliced_b = orig_b[curr_kept_idx]
            
            # Slice input channels if not the first layer
            if prev_kept_idx is not None:
                sliced_w = sliced_w[:, prev_kept_idx, :]
                
            narrow_model.features[i].weight.data = sliced_w
            narrow_model.features[i].bias.data = sliced_b
            
            prev_kept_idx = curr_kept_idx
            
        elif isinstance(model.features[i], nn.BatchNorm1d):
            # BN channels match the PREVIOUS Conv1d output channels
            orig_bn = model.features[i]
            sliced_bn = narrow_model.features[i]
            
            sliced_bn.weight.data = orig_bn.weight.data[prev_kept_idx]
            sliced_bn.bias.data = orig_bn.bias.data[prev_kept_idx]
            sliced_bn.running_mean.data = orig_bn.running_mean.data[prev_kept_idx]
            sliced_bn.running_var.data = orig_bn.running_var.data[prev_kept_idx]
            
    # Classifier block
    # Linear 1
    orig_fc1_w = model.classifier[linear1_idx].weight.data
    orig_fc1_b = model.classifier[linear1_idx].bias.data
    
    # Input comes from features output (AdaptiveAvgPool1d flattens it)
    sliced_fc1_w = orig_fc1_w[kept_fc1_idx][:, prev_kept_idx]
    sliced_fc1_b = orig_fc1_b[kept_fc1_idx]
    
    narrow_model.classifier[linear1_idx].weight.data = sliced_fc1_w
    narrow_model.classifier[linear1_idx].bias.data = sliced_fc1_b
    
    # Linear 2 (No output pruning, just input pruning from prev layer)
    orig_fc2_w = model.classifier[linear2_idx].weight.data
    orig_fc2_b = model.classifier[linear2_idx].bias.data
    
    sliced_fc2_w = orig_fc2_w[:, kept_fc1_idx]
    
    narrow_model.classifier[linear2_idx].weight.data = sliced_fc2_w
    narrow_model.classifier[linear2_idx].bias.data = orig_fc2_b
    
    # 4. Fine-tune the reconstructed model
    print("Narrow structured model rebuilt successfully. Fine-tuning for 5 epochs...")
    narrow_model.train()
    train_loader = get_train_loader(data_dir)
    optimizer = torch.optim.Adam(narrow_model.parameters(), lr=1e-4) # Fine-tune LR
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(5):
        epoch_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = narrow_model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/5 - Loss: {epoch_loss/len(train_loader):.4f}")
            
    narrow_model.eval()
    
    print("Structured Pruning completely applied. Evaluating narrow model...")
    evaluate_model(
        narrow_model, 
        model_name="ECGNet1D_Narrow_StructPruned", 
        technique_id="P2", 
        technique_name="Structured Pruning (30%)",
        save_path=save_metrics_path,
        device="cpu",
        save_model_path=save_model_path
    )

if __name__ == "__main__":
    main()
