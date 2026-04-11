"""
Phase 3: Performance Matrix Sweep
Runs all optimized models across 3 VM tiers in Docker containers
with enforced CPU/RAM limits, then collects results into a CSV.
"""
import subprocess
import os
import time
import json
import glob
import pandas as pd

COMPOSE_FILE = "environment/docker-compose.yml"
RESULTS_DIR = "results/phase3"

# All techniques to benchmark
TECHNIQUES = [
    {"tech_id": "B0",  "tech_name": "Baseline",             "path": "/app/results/baseline/baseline_best.pt"},
    {"tech_id": "Q1",  "tech_name": "Dynamic Quant",        "path": "/app/results/optimization/Q1_model.pt"},
    {"tech_id": "Q2",  "tech_name": "Static PTQ",           "path": "/app/results/optimization/Q2_model.pt"},
    {"tech_id": "Q3",  "tech_name": "QAT",                  "path": "/app/results/optimization/Q3_model.pt"},
    {"tech_id": "Q4",  "tech_name": "Weight-only FP16",     "path": "/app/results/optimization/Q4_model.pt"},
    {"tech_id": "Q5",  "tech_name": "Mixed Precision",      "path": "/app/results/optimization/Q5_model.pt"},
    {"tech_id": "P1",  "tech_name": "Unstructured Pruning", "path": "/app/results/optimization/P1_model.pt"},
    {"tech_id": "P2",  "tech_name": "Structured Pruning",   "path": "/app/results/optimization/P2_model.pt"},
    {"tech_id": "P3",  "tech_name": "Global Magnitude",     "path": "/app/results/optimization/P3_model.pt"},
]

# VM services defined in docker-compose.yml
VMS = ["vm1", "vm2", "vm3"]

def run_benchmark(vm_service, tech):
    """Run node_eval.py inside in the specified VM container."""
    tech_id   = tech["tech_id"]
    tech_name = tech["tech_name"]
    model_path = tech["path"]
    vm_id = vm_service.upper()

    print(f"\n  → [{vm_id}] Running {tech_id} ({tech_name})...")

    cmd = (
        f"docker compose -f {COMPOSE_FILE} run --rm "
        f"-e VM_ID={vm_id} "
        f"-e MODEL_PATH={model_path} "
        f"-e TECH_ID={tech_id} "
        f'-e TECH_NAME="{tech_name}" '
        f"-e COLLECTIVE_MODE=false "
        f"-e DATA_PATH=/mitbih/mitbih_test.csv "
        f"{vm_service}"
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    ✗ Error: {result.stderr[-500:]}")
    else:
        print(f"    ✓ Done")
    return result.returncode == 0

def collect_results():
    """Read all per-run JSON files and merge into a single DataFrame."""
    rows = []
    json_files = glob.glob(os.path.join(RESULTS_DIR, "*.json"))
    for path in sorted(json_files):
        try:
            with open(path) as f:
                rows.append(json.load(f))
        except Exception as e:
            print(f"  Warning: could not read {path}: {e}")

    if not rows:
        print("No result JSONs found.")
        return None

    df = pd.DataFrame(rows)
    # Reorder columns
    cols = ["vm_id","tech_id","tech_name","accuracy",
            "avg_latency_ms","std_latency_ms",
            "avg_cpu_percent","avg_ram_mb","model_path"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Clean up old per-run JSONs (keep the CSV)
    for f in glob.glob(os.path.join(RESULTS_DIR, "VM*_results.json")):
        os.remove(f)

    print("=" * 60)
    print("  Phase 3: Performance Matrix Sweep")
    print(f"  {len(VMS)} VMs × {len(TECHNIQUES)} techniques = {len(VMS)*len(TECHNIQUES)} runs")
    print("=" * 60)

    total = len(VMS) * len(TECHNIQUES)
    done  = 0
    for vm_service in VMS:
        print(f"\n[{vm_service.upper()}] Starting benchmark sweep...")
        for tech in TECHNIQUES:
            ok = run_benchmark(vm_service, tech)
            done += 1
            print(f"  Progress: {done}/{total}")
            if not ok:
                print(f"  ⚠ Skipping — check Docker logs for {vm_service}")

    print("\n" + "=" * 60)
    print("  Collecting results...")
    df = collect_results()
    if df is not None:
        out_csv = os.path.join(RESULTS_DIR, "phase3_full_matrix.csv")
        df.to_csv(out_csv, index=False)
        print(f"\n  ✓ Matrix saved → {out_csv}")
        print(f"\n{df[['vm_id','tech_id','accuracy','avg_latency_ms','avg_cpu_percent','avg_ram_mb']].to_string(index=False)}")
    else:
        print("  ✗ No results collected. Check the JSON files in results/phase3/")

    print("\n  → Next: python optimization/select_best.py")

if __name__ == "__main__":
    main()
