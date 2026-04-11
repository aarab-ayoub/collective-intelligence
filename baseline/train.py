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

# Add project root to path for modular imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.config import (
    SEED, DATA_DIR, BASELINE_DIR, GRAPHS_DIR, 
    BASELINE_MODEL_NAME, BASELINE_MODEL_PATH
)
from utils.data_loader import get_train_loader, get_test_loader
from models.ecg_net import ECGNet1D

# ─── Config ───────────────────────────────────────────────────────────────────
BATCH_SIZE = 256
EPOCHS = 30
PATIENCE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 2.0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

OUT_DIR = BASELINE_DIR
PLOTS_DIR = GRAPHS_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)
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
# We use custom split here to maintain the 85/15 ratio for baseline
train_df = pd.read_csv(DATA_DIR / "mitbih_train.csv", header=None)
test_df = pd.read_csv(DATA_DIR / "mitbih_test.csv", header=None)

X_trainval = train_df.iloc[:, :-1].values.astype(np.float32)
y_trainval = train_df.iloc[:, -1].values.astype(np.int64)
X_test = test_df.iloc[:, :-1].values.astype(np.float32)
y_test = test_df.iloc[:, -1].values.astype(np.int64)

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

# ─── Training setup ──────────────────────────────────────────────────────────
x_train_t = torch.tensor(X_train).unsqueeze(1)
y_train_t = torch.tensor(y_train)
x_val_t = torch.tensor(X_val).unsqueeze(1)
y_val_t = torch.tensor(y_val)
x_test_t = torch.tensor(X_test).unsqueeze(1)
y_test_t = torch.tensor(y_test)

counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float32)
total = counts.sum()
class_weights = total / (NUM_CLASSES * counts)
class_weights = np.clip(class_weights, 1.0, 10.0)
weight_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

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
    train_correct = 0
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
        train_correct += (logits.argmax(1) == by).sum().item()
    train_loss /= max(n_samples, 1)
    train_acc = train_correct / max(n_samples, 1)

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
        "train_accuracy": float(train_acc),
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
torch.save(model, BASELINE_MODEL_PATH)
print(f"\nSaved full model: {BASELINE_MODEL_PATH}")

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
model_size_bytes = os.path.getsize(BASELINE_MODEL_PATH)
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

# ─── Plots (one graph per file) ──────────────────────────────────────────────
epochs_list = [h["epoch"] for h in history]

# 1) Loss curves
plt.figure(figsize=(8, 5))
plt.plot(epochs_list, [h["train_loss"] for h in history], "b-o", label="Train Loss", markersize=3)
plt.plot(epochs_list, [h["val_loss"] for h in history], "r-o", label="Val Loss", markersize=3)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curves")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
loss_plot_path = PLOTS_DIR / "baseline_loss_curves.png"
plt.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
plt.close()

# 2) Accuracy graph (train evolution + final test accuracy reference)
plt.figure(figsize=(8, 5))
plt.plot(epochs_list, [h["train_accuracy"] for h in history], "g-o", label="Train Accuracy", markersize=3)
plt.axhline(y=test_acc, color="purple", linestyle="--", alpha=0.8, label=f"Test Accuracy ({test_acc:.4f})")
plt.scatter([epochs_list[-1]], [test_acc], color="purple", s=45)
plt.axhline(y=0.85, color="gray", linestyle="--", alpha=0.5, label="Acc target (85%)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train Accuracy vs Test Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
acc_plot_path = PLOTS_DIR / "baseline_accuracy_evolution.png"
plt.savefig(acc_plot_path, dpi=150, bbox_inches="tight")
# Keep legacy filename for compatibility with existing reports.
legacy_plot_path = PLOTS_DIR / "baseline_cnn_plots.png"
plt.savefig(legacy_plot_path, dpi=150, bbox_inches="tight")
plt.close()

# 3) Confusion matrix
plt.figure(figsize=(6.5, 5.5))
im = plt.imshow(cm, cmap="Blues")
plt.xticks(range(NUM_CLASSES), LABELS)
plt.yticks(range(NUM_CLASSES), LABELS)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=8)
plt.colorbar(im)
plt.tight_layout()
cm_plot_path = PLOTS_DIR / "baseline_confusion_matrix.png"
plt.savefig(cm_plot_path, dpi=150, bbox_inches="tight")
plt.close()

print(
    "Saved plots: "
    f"{loss_plot_path}, {acc_plot_path}, {cm_plot_path} "
    f"(legacy: {legacy_plot_path})"
)

print("\n✅ Phase 1 complete!")
