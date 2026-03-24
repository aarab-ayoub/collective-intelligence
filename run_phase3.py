import os
import subprocess
import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results" / "phase3"
OPTIMIZATION_DIR = PROJECT_ROOT / "results" / "optimization"
BASELINE_DIR = PROJECT_ROOT / "results" / "baseline"

TECHNIQUES = [
    {"id": "B0", "name": "Baseline", "path": "/app/results/baseline/baseline_best.pt"},
    {"id": "Q1", "name": "Dynamic Quant", "path": "/app/results/optimization/Q1_model.pt"},
    {"id": "Q2", "name": "Static PTQ", "path": "/app/results/optimization/Q2_model.pt"},
    {"id": "Q3", "name": "QAT", "path": "/app/results/optimization/Q3_model.pt"},
    {"id": "Q4", "name": "Weight-only FP16", "path": "/app/results/optimization/Q4_model.pt"},
    {"id": "Q5", "name": "Mixed Precision", "path": "/app/results/optimization/Q5_model.pt"},
    {"id": "P1", "name": "Unstructured Pruning", "path": "/app/results/optimization/P1_model.pt"},
    {"id": "P2", "name": "Structured Pruning", "path": "/app/results/optimization/P2_model.pt"},
    {"id": "P3", "name": "Global Magnitude", "path": "/app/results/optimization/P3_model.pt"},
]

VMS = ["vm1", "vm2", "vm3"]

def run_cmd(cmd):
    print(f"Exec: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result.stdout

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Building Docker containers...")
    run_cmd("docker compose -f deployment/docker-compose.yml build")

    matrix_rows = []

    for vm in VMS:
        print(f"\n>>> Evaluating all techniques on {vm.upper()}...")
        for tech in TECHNIQUES:
            print(f"Testing {tech['name']} ({tech['id']}) on {vm}...")
            
            # Run one-off container for each tech to ensure clean resource stats
            cmd = (
                f"docker compose -f deployment/docker-compose.yml run --rm "
                f"-e MODEL_PATH={tech['path']} "
                f"-e TECH_ID={tech['id']} "
                f"-e TECH_NAME='{tech['name']}' "
                f"{vm}"
            )
            run_cmd(cmd)
            
            # Load result
            res_path = RESULTS_DIR / f"{vm.upper()}_{tech['id']}_results.json"
            if res_path.exists():
                with open(res_path) as f:
                    matrix_rows.append(json.load(f))
            else:
                print(f"Warning: Result file {res_path} not found.")

    # Generate 3x8 Matrix (actually 3x9 with Baseline)
    if matrix_rows:
        df = pd.DataFrame(matrix_rows)
        pivot_acc = df.pivot(index='vm_id', columns='tech_id', values='accuracy')
        pivot_lat = df.pivot(index='vm_id', columns='tech_id', values='avg_latency_ms')
        
        print("\n" + "="*80)
        print("PHASE 3 PERFORMANCE MATRIX (LATENCY ms)")
        print("="*80)
        print(pivot_lat)
        print("\n" + "="*80)
        print("PHASE 3 PERFORMANCE MATRIX (ACCURACY)")
        print("="*80)
        print(pivot_acc)
        
        df.to_csv(RESULTS_DIR / "phase3_full_matrix.csv", index=False)
        pivot_lat.to_csv(RESULTS_DIR / "phase3_latency_matrix.csv")
        pivot_acc.to_csv(RESULTS_DIR / "phase3_accuracy_matrix.csv")
        
        print(f"\nFull results saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()
