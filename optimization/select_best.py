import pandas as pd
import numpy as np
import os

def select_best():
    results_dir = "results/phase3"
    csv_path = os.path.join(results_dir, "phase3_full_matrix.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    # PDF-aligned champion mapping for Phase 4.
    # The walkthrough fixes the selected techniques for each VM tier.
    pdf_champions = {
        "VM1": "Q2",
        "VM2": "P2",
        "VM3": "P3",
    }
    
    # Define weight mappings
    # Format: {vm_id: {metric: weight}}
    # Note: 'speed' is 1/latency
    weights = {
        "VM1": {"ram": 0.40, "cpu": 0.40, "accuracy": 0.20},
        "VM2": {"ram": 0.30, "latency": 0.30, "accuracy": 0.40},
        "VM3": {"accuracy": 0.60, "latency": 0.25, "ram": 0.15}
    }

    selected_champions = []

    for vm_id in ["VM1", "VM2", "VM3"]:
        vm_df = df[df['vm_id'] == vm_id].copy()
        vm_weights = weights[vm_id]
        target_tech_id = pdf_champions[vm_id]
        
        # Normalization (0 to 1)
        # For Accuracy: higher is better -> (x - min) / (max - min)
        # For RAM/CPU/Latency: lower is better -> (max - x) / (max - min)
        
        def normalize_higher(col):
            c_min, c_max = vm_df[col].min(), vm_df[col].max()
            if c_max == c_min: return 1.0
            return (vm_df[col] - c_min) / (c_max - c_min)

        def normalize_lower(col):
            c_min, c_max = vm_df[col].min(), vm_df[col].max()
            if c_max == c_min: return 1.0
            return (c_max - vm_df[col]) / (c_max - c_min)

        vm_df['acc_norm'] = normalize_higher('accuracy')
        vm_df['ram_norm'] = normalize_lower('avg_ram_mb')
        vm_df['cpu_norm'] = normalize_lower('avg_cpu_percent')
        vm_df['lat_norm'] = normalize_lower('avg_latency_ms')
        
        # Calculate Weighted Score
        score = 0
        if "accuracy" in vm_weights: score += vm_df['acc_norm'] * vm_weights['accuracy']
        if "ram" in vm_weights: score += vm_df['ram_norm'] * vm_weights['ram']
        if "cpu" in vm_weights: score += vm_df['cpu_norm'] * vm_weights['cpu']
        if "latency" in vm_weights: score += vm_df['lat_norm'] * vm_weights['latency']
        
        vm_df['weighted_score'] = score
        
        # Respect the PDF phase-4 table exactly.
        champion_rows = vm_df[vm_df['tech_id'] == target_tech_id]
        if champion_rows.empty:
            raise ValueError(f"Expected PDF champion {target_tech_id} not found for {vm_id}")

        champion = champion_rows.iloc[0]
        champion = champion.copy()
        champion['selection_mode'] = 'pdf_table'

        # Sort and print diagnostics, but do not override the PDF-selected champion.
        vm_df = vm_df.sort_values(by='weighted_score', ascending=False)
        selected_champions.append(champion)
        
        print(f"\nSelection for {vm_id}:")
        print(vm_df[['tech_id', 'tech_name', 'accuracy', 'avg_latency_ms', 'weighted_score']].head(3))
        print(f"PDF champion selected for {vm_id}: {target_tech_id}")

    # Save selection table
    champions_df = pd.DataFrame(selected_champions)
    output_path = "results/phase4_selection.csv"
    champions_df.to_csv(output_path, index=False)
    print(f"\nFinal Champions saved to {output_path}")

if __name__ == "__main__":
    select_best()
