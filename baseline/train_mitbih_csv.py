"""
Phase 1 — Train Baseline 1D-CNN on MIT-BIH CSV Dataset
=======================================================
Dataset: mitbih_train.csv / mitbih_test.csv (Kaggle preprocessed)
  - 187 columns: 186 ECG features (normalized heartbeat) + 1 label (0-4)
  - Classes: N(0), S(1), V(2), F(3), Q(4)  [AAMI mapping]
  - Train: 87,554 samples, Test: 21,892 samples

Architecture: Compact 1D-CNN (~55K params) designed for embedded deployment.
"""

import os, sys, json, time, random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ─── Config ───────────────────────────────────────────────────────────────────
SEED = 42
BATCH_SIZE = 256
EPOCHS = 30
PATIENCE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 2.0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT.parent / "mitbih"
OUT_DIR = PROJECT_ROOT / "results" / "baseline"
WEIGHTS_DIR = PROJECT_ROOT / "results" / "baseline"
PLOTS_DIR = PROJECT_ROOT / "results" / "graphs_and_images"
OUT_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["N", "S", "V", "F", "Q"]
NUM_CLASSES = 5

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
)
print(f"Device: {device}")

# ─── Data loading ─────────────────────────────────────────────────────────────
print("Loading data...")
train_df = pd.read_csv(DATA_DIR / "mitbih_train.csv", header=None)
test_df = pd.read_csv(DATA_DIR / "mitbih_test.csv", header=None)

X_trainval = train_df.iloc[:, :-1].values.astype(np.float32)
y_trainval = train_df.iloc[:, -1].values.astype(np.int64)
X_test = test_df.iloc[:, :-1].values.astype(np.float32)
y_test = test_df.iloc[:, -1].values.astype(np.int64)

# Stratified train/val split (85/15)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15, random_state=SEED, stratify=y_trainval
)

def print_dist(name, y):
    cnt = np.bincount(y, minlength=NUM_CLASSES)
    total = len(y)
    print(f"\n{name} (n={total}):")
    for i in range(NUM_CLASSES):
        print(f"  {LABELS[i]}: {cnt[i]:6d} ({100*cnt[i]/total:5.2f}%)")

print_dist("TRAIN", y_train)
print_dist("VAL", y_val)
print_dist("TEST", y_test)

# ─── Model ────────────────────────────────────────────────────────────────────
# Import from the absolute package now that utils is initialized
sys.path.append(str(PROJECT_ROOT))
from utils.utils import ECGNet1D

# ─── Training setup ──────────────────────────────────────────────────────────
# Convert to tensors (add channel dim for Conv1d)
x_train_t = torch.tensor(X_train).unsqueeze(1)  # (N, 1, 186)
y_train_t = torch.tensor(y_train)
x_val_t = torch.tensor(X_val).unsqueeze(1)
y_val_t = torch.tensor(y_val)
x_test_t = torch.tensor(X_test).unsqueeze(1)
y_test_t = torch.tensor(y_test)

# Class weights for CrossEntropy (moderate, sqrt of inverse frequency)
counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float32)
total = counts.sum()
# Use effective number of samples weighting (moderate)
class_weights = total / (NUM_CLASSES * counts)
# Cap weights to avoid too aggressive rebalancing  
class_weights = np.clip(class_weights, 1.0, 10.0)
weight_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
print(f"\nCross-entropy weights: {dict(zip(LABELS, [round(float(w), 3) for w in class_weights]))}")

train_loader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(x_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(x_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)

model = ECGNet1D().to(device)
criterion = nn.CrossEntropyLoss(weight=weight_t)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

# ─── Training loop ───────────────────────────────────────────────────────────
print(f"\nTraining for up to {EPOCHS} epochs (patience={PATIENCE})...\n")
history = []
best_f1 = -1.0
best_state = None
wait = 0

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    train_loss = 0.0
    n_samples = 0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        logits = model(bx)
        loss = criterion(logits, by)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        train_loss += loss.item() * bx.size(0)
        n_samples += bx.size(0)
    train_loss /= max(n_samples, 1)

    # Validate
    model.eval()
    val_preds, val_targs = [], []
    val_loss = 0.0
    n_val = 0
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)
            loss = criterion(logits, by)
            val_loss += loss.item() * bx.size(0)
            n_val += bx.size(0)
            val_preds.extend(logits.argmax(1).cpu().numpy())
            val_targs.extend(by.cpu().numpy())
    val_loss /= max(n_val, 1)

    val_acc = accuracy_score(val_targs, val_preds)
    val_f1 = f1_score(val_targs, val_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(val_targs, val_preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)

    history.append({
        "epoch": epoch,
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "val_accuracy": float(val_acc),
        "val_macro_f1": float(val_f1),
    })

    pc_str = " ".join(f"{LABELS[i]}={per_class_f1[i]:.3f}" for i in range(NUM_CLASSES))
    print(f"E{epoch:02d}  loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
          f"val_acc={val_acc:.4f}  val_f1={val_f1:.4f}  [{pc_str}]")

    scheduler.step()

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

# ─── Test evaluation ─────────────────────────────────────────────────────────
if best_state is None:
    raise RuntimeError("No valid checkpoint captured!")

model.load_state_dict(best_state)
model.eval()

# Save weights
weight_path = WEIGHTS_DIR / "baseline_best.pt"
torch.save(model, weight_path)
print(f"\nSaved full model: {weight_path}")

# Test inference
test_preds, test_targs = [], []
t0 = time.perf_counter()
with torch.no_grad():
    for bx, by in test_loader:
        bx = bx.to(device)
        logits = model(bx)
        test_preds.extend(logits.argmax(1).cpu().numpy())
        test_targs.extend(by.numpy())
inference_time = (time.perf_counter() - t0) * 1000  # total ms
avg_inference_ms = inference_time / len(test_targs)

test_preds = np.array(test_preds)
test_targs = np.array(test_targs)

test_acc = accuracy_score(test_targs, test_preds)
test_f1 = f1_score(test_targs, test_preds, average="macro", zero_division=0)
per_class = f1_score(test_targs, test_preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
per_class_prec = precision_score(test_targs, test_preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
per_class_rec = recall_score(test_targs, test_preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
cm = confusion_matrix(test_targs, test_preds, labels=list(range(NUM_CLASSES)))

# Model size
model_size_bytes = os.path.getsize(weight_path)
model_size_mb = model_size_bytes / (1024 * 1024)

print("\n" + "=" * 60)
print("PHASE 1 — BASELINE TEST RESULTS")
print("=" * 60)
print(f"Test accuracy     : {100*test_acc:.2f}%")
print(f"Test macro F1     : {test_f1:.4f}")
print(f"Per-class F1      : {dict(zip(LABELS, [round(float(f), 4) for f in per_class]))}")
print(f"Per-class Recall  : {dict(zip(LABELS, [round(float(r), 4) for r in per_class_rec]))}")
print(f"Model size        : {model_size_mb:.2f} MB")
print(f"Model parameters  : {n_params:,}")
print(f"Avg inference time: {avg_inference_ms:.4f} ms/sample")
print(f"\nConfusion Matrix:\n{cm}")
print(f"\n{classification_report(test_targs, test_preds, target_names=LABELS, zero_division=0)}")

ok = test_acc > 0.85 and test_f1 > 0.75
print("✅ BASELINE CRITERIA MET" if ok else "❌ CRITERIA NOT MET")

# ─── Save metrics ────────────────────────────────────────────────────────────
metrics = {
    "dataset": "mitbih_csv",
    "seed": SEED,
    "model_architecture": "ECGNet1D",
    "n_params": n_params,
    "test_accuracy": float(test_acc),
    "test_macro_f1": float(test_f1),
    "per_class_f1": dict(zip(LABELS, [float(f) for f in per_class])),
    "per_class_precision": dict(zip(LABELS, [float(p) for p in per_class_prec])),
    "per_class_recall": dict(zip(LABELS, [float(r) for r in per_class_rec])),
    "confusion_matrix": cm.tolist(),
    "model_size_mb": float(model_size_mb),
    "avg_inference_time_ms": float(avg_inference_ms),
    "history": history,
    "literature_comparison": [
        {"paper": "Kachuee et al. (2018)", "accuracy": 0.934, "note": "1D-CNN on same dataset"},
        {"paper": "Acharya et al. (2017)", "accuracy": 0.942, "note": "9-layer CNN"},
        {"paper": "Hannun et al. (2019)", "accuracy": 0.970, "note": "34-layer ResNet (much larger)"},
    ],
}

metrics_path = OUT_DIR / "baseline_cnn_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nSaved metrics: {metrics_path}")

# ─── Plots ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Training curves
epochs_list = [h["epoch"] for h in history]
axes[0].plot(epochs_list, [h["train_loss"] for h in history], "b-o", label="Train Loss", markersize=3)
axes[0].plot(epochs_list, [h["val_loss"] for h in history], "r-o", label="Val Loss", markersize=3)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_title("Loss Curves")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_list, [h["val_accuracy"] for h in history], "g-o", label="Val Accuracy", markersize=3)
axes[1].plot(epochs_list, [h["val_macro_f1"] for h in history], "m-o", label="Val Macro F1", markersize=3)
axes[1].axhline(y=0.85, color="gray", linestyle="--", alpha=0.5, label="Acc target (85%)")
axes[1].axhline(y=0.75, color="orange", linestyle="--", alpha=0.5, label="F1 target (0.75)")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score"); axes[1].set_title("Accuracy & F1")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

# Confusion matrix
im = axes[2].imshow(cm, cmap="Blues")
axes[2].set_xticks(range(NUM_CLASSES)); axes[2].set_xticklabels(LABELS)
axes[2].set_yticks(range(NUM_CLASSES)); axes[2].set_yticklabels(LABELS)
axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("True"); axes[2].set_title("Confusion Matrix")
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        axes[2].text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=8)
plt.colorbar(im, ax=axes[2])

plt.tight_layout()
plot_path = PLOTS_DIR / "baseline_cnn_plots.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved plots: {plot_path}")

print("\n✅ Phase 1 complete!")
