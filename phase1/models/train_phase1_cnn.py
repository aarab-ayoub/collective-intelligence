import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler


class TinyECGCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.3, base_channels: int = 24):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        self.features = nn.Sequential(
            nn.Conv1d(12, c1, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(c1),
            nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(c2),
            nn.MaxPool1d(2),
            nn.Conv1d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(c3, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.squeeze(-1)
        return self.classifier(x)


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class ResECGNet(nn.Module):
    def __init__(self, num_classes: int, base_channels: int = 24, dropout: float = 0.3):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.stem = nn.Sequential(
            nn.Conv1d(12, c1, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            ResidualBlock1D(c1, c1, stride=1),
            ResidualBlock1D(c1, c1, stride=1),
        )
        self.layer2 = nn.Sequential(
            ResidualBlock1D(c1, c2, stride=2),
            ResidualBlock1D(c2, c2, stride=1),
        )
        self.layer3 = nn.Sequential(
            ResidualBlock1D(c2, c3, stride=2),
            ResidualBlock1D(c3, c3, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(c3, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


def build_model(model_name: str, num_classes: int, dropout: float, base_channels: int) -> nn.Module:
    if model_name == "resnet":
        return ResECGNet(num_classes=num_classes, base_channels=base_channels, dropout=dropout)
    return TinyECGCNN(num_classes=num_classes, dropout=dropout, base_channels=base_channels)


def load_split(data_dir: Path, split: str):
    x = np.load(data_dir / f"X_{split}.npy")
    y = np.load(data_dir / f"y_{split}.npy")
    x = np.transpose(x, (0, 2, 1)).astype(np.float32)
    y = y.astype(np.int64)
    return x, y


def macro_f1_from_preds(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    f1_scores = []
    for class_id in range(num_classes):
        tp = np.sum((y_true == class_id) & (y_pred == class_id))
        fp = np.sum((y_true != class_id) & (y_pred == class_id))
        fn = np.sum((y_true == class_id) & (y_pred != class_id))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if (precision + recall) == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))

    return float(np.mean(f1_scores)) if f1_scores else 0.0


def evaluate(model, loader, criterion, device, num_classes: int):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())

    if all_preds:
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        f1_macro = macro_f1_from_preds(y_true, y_pred, num_classes=num_classes)
    else:
        f1_macro = 0.0

    return total_loss / max(total, 1), correct / max(total, 1), f1_macro


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ECGAugmentDataset(Dataset):
    def __init__(
        self,
        x_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        noise_std: float = 0.01,
        scale_jitter: float = 0.1,
        time_mask_max: int = 40,
    ):
        self.x_tensor = x_tensor
        self.y_tensor = y_tensor
        self.noise_std = noise_std
        self.scale_jitter = scale_jitter
        self.time_mask_max = time_mask_max

    def __len__(self):
        return self.x_tensor.shape[0]

    def __getitem__(self, idx):
        x = self.x_tensor[idx].clone()
        y = self.y_tensor[idx]

        if self.scale_jitter > 0 and np.random.rand() < 0.8:
            scale = 1.0 + np.random.uniform(-self.scale_jitter, self.scale_jitter)
            x = x * float(scale)

        if self.noise_std > 0 and np.random.rand() < 0.8:
            x = x + torch.randn_like(x) * self.noise_std

        if self.time_mask_max > 0 and np.random.rand() < 0.5:
            seq_len = x.shape[1]
            width = np.random.randint(1, min(self.time_mask_max, seq_len) + 1)
            start = np.random.randint(0, seq_len - width + 1)
            x[:, start : start + width] = 0.0

        return x, y


def compute_train_norm_stats(x_train: np.ndarray):
    mean = np.mean(x_train, axis=(0, 1), keepdims=True)
    std = np.std(x_train, axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_normalization(x: np.ndarray, mean: np.ndarray, std: np.ndarray, clip_value: float):
    x_norm = (x - mean) / std
    if clip_value > 0:
        x_norm = np.clip(x_norm, -clip_value, clip_value)
    return x_norm.astype(np.float32)


def build_weighted_sampler(y_tensor: torch.Tensor):
    y_np = y_tensor.numpy()
    class_counts = np.bincount(y_np)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_np]
    sample_weights = torch.from_numpy(sample_weights).double()
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def prepare_data(
    data_dir: Path,
    batch_size: int,
    normalize_inputs: bool = True,
    clip_value: float = 5.0,
    use_weighted_sampler: bool = False,
):
    required = [
        data_dir / "X_train.npy",
        data_dir / "y_train.npy",
        data_dir / "X_val.npy",
        data_dir / "y_val.npy",
        data_dir / "X_test.npy",
        data_dir / "y_test.npy",
        data_dir / "label_map.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing processed files. Run Phase 0 notebook first. Missing: " + ", ".join(missing)
        )

    with open(data_dir / "label_map.json", "r", encoding="utf-8") as f:
        label_map = json.load(f)

    x_train, y_train = load_split(data_dir, "train")
    x_val, y_val = load_split(data_dir, "val")
    x_test, y_test = load_split(data_dir, "test")

    if normalize_inputs:
        mean, std = compute_train_norm_stats(x_train)
        x_train = apply_normalization(x_train, mean, std, clip_value=clip_value)
        x_val = apply_normalization(x_val, mean, std, clip_value=clip_value)
        x_test = apply_normalization(x_test, mean, std, clip_value=clip_value)

    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train)
    x_val_t = torch.from_numpy(x_val)
    y_val_t = torch.from_numpy(y_val)
    x_test_t = torch.from_numpy(x_test)
    y_test_t = torch.from_numpy(y_test)

    train_ds = ECGAugmentDataset(
        x_tensor=x_train_t,
        y_tensor=y_train_t,
        noise_std=0.01,
        scale_jitter=0.12,
        time_mask_max=40,
    )
    train_eval_ds = TensorDataset(x_train_t, y_train_t)
    val_ds = TensorDataset(x_val_t, y_val_t)
    test_ds = TensorDataset(x_test_t, y_test_t)

    if use_weighted_sampler:
        sampler = build_weighted_sampler(y_train_t)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    info = {
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "num_classes": len(label_map),
        "normalize_inputs": normalize_inputs,
        "clip_value": clip_value,
        "use_weighted_sampler": use_weighted_sampler,
    }

    return train_loader, train_eval_loader, val_loader, test_loader, label_map, info


def compute_class_weights(train_eval_loader, num_classes: int) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float64)
    for _, yb in train_eval_loader:
        y_np = yb.numpy()
        binc = np.bincount(y_np, minlength=num_classes)
        counts += binc

    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    weights = inv / np.mean(inv)
    return torch.tensor(weights, dtype=torch.float32)


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if torch.is_floating_point(v)
        }
        self.backup = {}

    def update(self, model: nn.Module):
        with torch.no_grad():
            state = model.state_dict()
            for key, shadow_val in self.shadow.items():
                shadow_val.mul_(self.decay).add_(state[key], alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        state = model.state_dict()
        for key, shadow_val in self.shadow.items():
            self.backup[key] = state[key].detach().clone()
            state[key].copy_(shadow_val)

    def restore(self, model: nn.Module):
        if not self.backup:
            return
        state = model.state_dict()
        for key, backup_val in self.backup.items():
            state[key].copy_(backup_val)
        self.backup = {}


def train(
    args,
    train_loader,
    train_eval_loader,
    val_loader,
    num_classes: int,
    out_dir: Path,
    device: torch.device,
):
    model = build_model(
        model_name=args.model,
        num_classes=num_classes,
        dropout=args.dropout,
        base_channels=args.base_channels,
    ).to(device)
    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(train_eval_loader, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(args.epochs, 1),
            eta_min=args.min_lr,
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
        )

    ema = None
    if args.ema_decay > 0:
        ema = ExponentialMovingAverage(model, decay=args.ema_decay)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_val_f1 = 0.0
    best_epoch = 0
    no_improve_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            if args.mixup_alpha > 0:
                mixed_x, y_a, y_b, lam = mixup_batch(xb, yb, args.mixup_alpha)
                logits = model(mixed_x)
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            else:
                logits = model(xb)
                loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update(model)

        use_ema_now = ema is not None and epoch >= args.ema_start_epoch

        if use_ema_now:
            ema.apply_shadow(model)

        train_loss, train_acc, train_f1 = evaluate(
            model, train_eval_loader, criterion, device, num_classes=num_classes
        )
        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, criterion, device, num_classes=num_classes
        )

        if use_ema_now:
            ema.restore(model)

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "train_f1": float(train_f1),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "val_f1": float(val_f1),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        if args.scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_loss)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_loss < (best_val_loss - args.min_delta):
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_epoch = epoch
            no_improve_epochs = 0
            if use_ema_now:
                ema.apply_shadow(model)
                torch.save(model.state_dict(), out_dir / "best_model.pt")
                ema.restore(model)
            else:
                torch.save(model.state_dict(), out_dir / "best_model.pt")
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= args.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch} "
                f"(best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f})"
            )
            break

    return model, criterion, history, best_val_acc, best_val_loss, best_val_f1, best_epoch


def evaluate_checkpoint(args):
    data_dir = Path(args.data_dir)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    _, _, _, test_loader, label_map, info = prepare_data(
        data_dir=data_dir,
        batch_size=args.batch_size,
        normalize_inputs=args.normalize_inputs,
        clip_value=args.input_clip,
        use_weighted_sampler=False,
    )
    device = get_device()
    num_classes = len(label_map)

    model = build_model(
        model_name=args.model,
        num_classes=num_classes,
        dropout=args.dropout,
        base_channels=args.base_channels,
    ).to(device)

    class_weights = None
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    test_loss, test_acc, test_f1 = evaluate(
        model, test_loader, criterion, device, num_classes=num_classes
    )

    print(
        f"Eval-only | device={device.type} | n_test={info['n_test']} | "
        f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} test_f1={test_f1:.4f}"
    )


def run(args):
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, train_eval_loader, val_loader, test_loader, label_map, info = prepare_data(
        data_dir=data_dir,
        batch_size=args.batch_size,
        normalize_inputs=args.normalize_inputs,
        clip_value=args.input_clip,
        use_weighted_sampler=args.use_weighted_sampler,
    )
    device = get_device()
    num_classes = len(label_map)

    print(
        f"Using device={device.type} | "
        f"n_train={info['n_train']} n_val={info['n_val']} n_test={info['n_test']} | "
        f"num_classes={info['num_classes']}"
    )

    model, criterion, history, best_val_acc, best_val_loss, best_val_f1, best_epoch = train(
        args=args,
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        out_dir=out_dir,
        device=device,
    )

    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
    test_loss, test_acc, test_f1 = evaluate(
        model, test_loader, criterion, device, num_classes=num_classes
    )

    print(
        f"Final test (best checkpoint) | "
        f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} test_f1={test_f1:.4f}"
    )

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    summary = {
        "device": device.type,
        "n_train": int(info["n_train"]),
        "n_val": int(info["n_val"]),
        "n_test": int(info["n_test"]),
        "num_classes": int(info["num_classes"]),
        "model": args.model,
        "scheduler": args.scheduler,
        "ema_decay": float(args.ema_decay),
        "mixup_alpha": float(args.mixup_alpha),
        "normalize_inputs": bool(args.normalize_inputs),
        "input_clip": float(args.input_clip),
        "use_weighted_sampler": bool(args.use_weighted_sampler),
        "epochs_requested": int(args.epochs),
        "epochs_ran": int(len(history)),
        "dropout": float(args.dropout),
        "weight_decay": float(args.weight_decay),
        "label_smoothing": float(args.label_smoothing),
        "early_stopping_patience": int(args.early_stopping_patience),
        "lr_patience": int(args.lr_patience),
        "lr_factor": float(args.lr_factor),
        "min_lr": float(args.min_lr),
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "best_val_f1": float(best_val_f1),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_f1": float(test_f1),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Best validation epoch: {best_epoch}")
    print(f"Saved best model to: {out_dir / 'best_model.pt'}")


def build_parser():
    parser = argparse.ArgumentParser(description="Phase 1 CNN baseline for PTB-XL")
    parser.add_argument("--data-dir", type=str, default="dataset/processed")
    parser.add_argument("--out-dir", type=str, default="results/phase1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--model", type=str, default="tiny", choices=["tiny", "resnet"])
    parser.add_argument("--normalize-inputs", action="store_true")
    parser.add_argument("--input-clip", type=float, default=5.0)
    parser.add_argument("--use-weighted-sampler", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--weight-decay", type=float, default=2e-3)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.08)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--scheduler", type=str, default="plateau", choices=["plateau", "cosine"])
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--ema-start-epoch", type=int, default=8)
    parser.add_argument("--early-stopping-patience", type=int, default=7)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.eval_only:
        evaluate_checkpoint(args)
    else:
        run(args)
