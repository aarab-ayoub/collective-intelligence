"""
preprocess.py — Prepare the PTB-XL ECG dataset for training.

PTB-XL: 21,837 clinical 12-lead ECG recordings, 10 s at 500 Hz.
Source : https://physionet.org/content/ptb-xl/1.0.3/

Steps:
  1. Load the PTB-XL metadata CSV (ptbxl_database.csv).
  2. Map the diagnostic superclass to a clean integer label.
  3. Load raw ECG signals via wfdb (500 Hz or 100 Hz resampled).
  4. Apply bandpass filtering (0.5–40 Hz) and z-score normalisation per lead.
  5. Split into train (70 %) / val (15 %) / test (15 %) with stratification.
  6. Save as NumPy arrays (.npy) and a metadata CSV per split.

Diagnostic superclasses (5 classes):
  NORM  — Normal ECG
  MI    — Myocardial Infarction
  STTC  — ST/T Change
  CD    — Conduction Disturbance
  HYP   — Hypertrophy

Usage:
    python preprocess.py [--raw_dir raw/ptb-xl] [--out_dir processed] [--seed 42]
                         [--sampling_rate 100]
"""

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Superclass mapping (from scp_statements.csv)
# ---------------------------------------------------------------------------

SUPERCLASS_LABELS = {
    "NORM": 0,
    "MI":   1,
    "STTC": 2,
    "CD":   3,
    "HYP":  4,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PTB-XL preprocessing")
    parser.add_argument("--raw_dir",       default="raw/ptb-xl",
                        help="Root of the downloaded PTB-XL folder")
    parser.add_argument("--out_dir",       default="processed")
    parser.add_argument("--sampling_rate", type=int, default=100,
                        choices=[100, 500],
                        help="Use 100 Hz records (faster) or 500 Hz (full resolution)")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--splits",        nargs=3, type=float,
                        default=[0.70, 0.15, 0.15],
                        metavar=("TRAIN", "VAL", "TEST"))
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

def bandpass_filter(signal: np.ndarray, fs: int,
                    low: float = 0.5, high: float = 40.0) -> np.ndarray:
    """Apply a 4th-order Butterworth bandpass filter to all 12 leads."""
    nyq = fs / 2.0
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal, axis=0)


def normalize(signal: np.ndarray) -> np.ndarray:
    """Z-score normalise each lead independently."""
    mean = signal.mean(axis=0, keepdims=True)
    std  = signal.std(axis=0, keepdims=True) + 1e-8
    return (signal - mean) / std


def load_ecg(record_path: str, fs: int) -> np.ndarray:
    """Load a single ECG record and return shape (samples, 12)."""
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal.astype(np.float32)   # (samples, 12)
    signal = bandpass_filter(signal, fs)
    signal = normalize(signal)
    return signal


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------

def extract_superclass(scp_codes_str: str,
                       agg_df: pd.DataFrame) -> str | None:
    """Return the dominant diagnostic superclass for a record."""
    try:
        codes = ast.literal_eval(scp_codes_str)
    except Exception:
        return None

    # Filter to diagnostic codes only
    diag_codes = {k: v for k, v in codes.items()
                  if k in agg_df.index and agg_df.loc[k, "diagnostic"] == 1}
    if not diag_codes:
        return None

    # Highest-likelihood diagnostic code
    best_code = max(diag_codes, key=diag_codes.get)
    superclass = agg_df.loc[best_code, "diagnostic_class"]
    return superclass if superclass in SUPERCLASS_LABELS else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    assert abs(sum(args.splits) - 1.0) < 1e-6, "Splits must sum to 1"

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fs = args.sampling_rate
    record_subdir = "records100" if fs == 100 else "records500"

    # Load metadata
    meta_path = raw_dir / "ptbxl_database.csv"
    agg_path  = raw_dir / "scp_statements.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}. Run download.sh first.")

    meta_df = pd.read_csv(meta_path, index_col="ecg_id")
    agg_df  = pd.read_csv(agg_path,  index_col=0)

    # Assign labels
    meta_df["superclass"] = meta_df["scp_codes"].apply(
        lambda s: extract_superclass(s, agg_df)
    )
    meta_df = meta_df.dropna(subset=["superclass"])
    meta_df["label"] = meta_df["superclass"].map(SUPERCLASS_LABELS)

    print(f"Total labelled records : {len(meta_df)}")
    print("Class distribution:")
    print(meta_df["superclass"].value_counts().to_string(), "\n")

    # Stratified split
    train_ratio, val_ratio, _ = args.splits
    train_idx, rest_idx = train_test_split(
        meta_df.index, train_size=train_ratio,
        stratify=meta_df["label"], random_state=args.seed
    )
    val_size_rel = val_ratio / (1.0 - train_ratio)
    val_idx, test_idx = train_test_split(
        rest_idx, train_size=val_size_rel,
        stratify=meta_df.loc[rest_idx, "label"], random_state=args.seed
    )

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    for split_name, idx in splits.items():
        split_df = meta_df.loc[idx]
        signals, labels = [], []

        for ecg_id, row in tqdm(split_df.iterrows(), total=len(split_df),
                                 desc=f"Loading {split_name}"):
            rel_path = row["filename_lr"] if fs == 100 else row["filename_hr"]
            record_path = str(raw_dir / rel_path).replace(".hea", "")
            try:
                sig = load_ecg(record_path, fs)
                signals.append(sig)
                labels.append(int(row["label"]))
            except Exception as e:
                print(f"[WARN] Skipping {ecg_id}: {e}")

        X = np.stack(signals).astype(np.float32)  # (N, samples, 12)
        y = np.array(labels, dtype=np.int64)

        split_out = out_dir / split_name
        split_out.mkdir(parents=True, exist_ok=True)
        np.save(split_out / "X.npy", X)
        np.save(split_out / "y.npy", y)
        split_df.to_csv(split_out / "metadata.csv")
        print(f"  {split_name:5s}: {len(X):5d} records  shape={X.shape}")

    # Save global stats
    stats = {
        "dataset": "PTB-XL",
        "sampling_rate_hz": fs,
        "n_leads": 12,
        "classes": {v: k for k, v in SUPERCLASS_LABELS.items()},
        "split_sizes": {s: len(i) for s, i in splits.items()},
        "seed": args.seed,
    }
    (out_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\nStats saved to {out_dir / 'dataset_stats.json'}")


if __name__ == "__main__":
    main()
