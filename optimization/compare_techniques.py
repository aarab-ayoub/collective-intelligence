import os
import sys
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Modular Imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.config import RESULTS_DIR, OPTIMIZATION_DIR, BASELINE_DIR

def main():
    # Baseline metrics
    baseline_metrics_file = BASELINE_DIR / "baseline_cnn_metrics.json"
    if baseline_metrics_file.exists():
        with open(baseline_metrics_file) as f:
            bm = json.load(f)
            baseline = {
                "id": "B0",
                "technique": "Baseline (FP32)",
                "accuracy": bm["test_accuracy"],
                "macro_f1": bm["test_macro_f1"],
                "size_mb": bm["model_size_mb"],
                "inference_ms": bm["avg_inference_time_ms"]
            }
    else:
        baseline = None
        
    # Read optimization metrics
    records = []
    if baseline:
        records.append(baseline)
        
    for p in OPTIMIZATION_DIR.glob("*_metrics.json"):
        with open(p) as f:
            records.append(json.load(f))
            
    if not records:
        print("No optimization metrics found!")
        return
        
    df = pd.DataFrame(records)
    # Sort by ID (B0, P1, P2, P3, Q1, Q2, Q3, Q4, Q5)
    df = df.sort_values(by="id")
    
    print("\n" + "="*80)
    print(f"{'ID':<4} | {'Technique':<32} | {'Accuracy':<10} | {'Macro F1':<10} | {'Size (MB)':<10} | {'Speed (ms/smpl)'}")
    print("-" * 80)
    for _, row in df.iterrows():
        print(f"{row['id']:<4} | {row['technique']:<32} | {row['accuracy']*100:>8.2f}% | {row['macro_f1']:>8.4f} | {row['size_mb']:>9.3f} | {row['inference_ms']:>8.4f}")
    print("="*80 + "\n")
    
    # Save CSV
    csv_path = OPTIMIZATION_DIR / "optimization_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV comparison to: {csv_path}")
    
    # Generate Plots (one graph per file)

    # Plot 1: Size vs Accuracy
    plt.figure(figsize=(8, 6))
    for _, row in df.iterrows():
        color = "red" if row['id'] == "B0" else ("blue" if row['id'].startswith("Q") else "green")
        marker = "*" if row['id'] == "B0" else "o"
        size = 200 if row['id'] == "B0" else 100
        plt.scatter(row['size_mb'], row['accuracy']*100, label=f"{row['id']} ({row['technique']})", color=color, s=size, marker=marker)
        plt.annotate(row['id'], (row['size_mb'], row['accuracy']*100), xytext=(5, 5), textcoords='offset points', fontsize=9)
        
    plt.xlabel("Model Size (MB)")
    plt.ylabel("Accuracy (%)")
    plt.title("Size vs Accuracy")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.gca().invert_xaxis()
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    size_plot_path = OPTIMIZATION_DIR / "optimization_size_vs_accuracy.png"
    plt.savefig(size_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    # Plot 2: Speed vs Accuracy
    plt.figure(figsize=(8, 6))
    for _, row in df.iterrows():
        color = "red" if row['id'] == "B0" else ("blue" if row['id'].startswith("Q") else "green")
        marker = "*" if row['id'] == "B0" else "o"
        size = 200 if row['id'] == "B0" else 100
        plt.scatter(row['inference_ms'], row['accuracy']*100, label=f"{row['id']} ({row['technique']})", color=color, s=size, marker=marker)
        plt.annotate(row['id'], (row['inference_ms'], row['accuracy']*100), xytext=(5, 5), textcoords='offset points', fontsize=9)
        
    plt.xlabel("Inference Time (ms/sample)")
    plt.ylabel("Accuracy (%)")
    plt.title("Speed vs Accuracy")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.gca().invert_xaxis()
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    speed_plot_path = OPTIMIZATION_DIR / "optimization_speed_vs_accuracy.png"
    plt.savefig(speed_plot_path, dpi=150, bbox_inches="tight")

    # Keep legacy filename for compatibility with existing docs/references.
    legacy_plot_path = OPTIMIZATION_DIR / "optimization_pareto_front.png"
    plt.savefig(legacy_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(
        "Saved optimization plots: "
        f"{size_plot_path}, {speed_plot_path} "
        f"(legacy: {legacy_plot_path})"
    )

if __name__ == "__main__":
    main()
