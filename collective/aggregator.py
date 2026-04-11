import os
import time
import json
import numpy as np
import pandas as pd
from collections import Counter

# Configuration from Phase 4 results
NODES = {
    "VM1": {"weight": 0.9748, "results_file": "VM1_pred.json"},
    "VM2": {"weight": 0.9773, "results_file": "VM2_pred.json"},
    "VM3": {"weight": 0.9821, "results_file": "VM3_pred.json"}
}

SHARED_DIR = os.getenv("SHARED_DIR", "/app/results/phase5")
CONFIDENCE_THRESHOLD = 0.70

def wait_for_predictions(specimen_id):
    """Wait for all nodes to output their prediction for a given specimen."""
    preds = {}
    start_time = time.time()
    timeout = 30 # seconds
    
    while len(preds) < 3 and (time.time() - start_time) < timeout:
        for node_id, info in NODES.items():
            if node_id not in preds:
                file_path = os.path.join(SHARED_DIR, f"specimen_{specimen_id}_{info['results_file']}")
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r') as f:
                            preds[node_id] = json.load(f)
                    except:
                        pass
        time.sleep(0.1)
    return preds

def solve_collective(specimen_id):
    print(f"\n[Aggregator] Solving Specimen {specimen_id}...")
    node_data = wait_for_predictions(specimen_id)
    
    if len(node_data) < 3:
        print(f"[Aggregator] Error: Timeout waiting for nodes for specimen {specimen_id}")
        return None

    # 1. Check Load Balancing / Resource Overload
    for node_id, data in node_data.items():
        if data.get('cpu_percent', 0) > 85 or data.get('ram_percent', 0) > 90:
            print(f"[Aggregator] WARNING: {node_id} is overloaded! CPU: {data['cpu_percent']}%, RAM: {data['ram_percent']}%")
            # In a real system, the orchestrator would redirect. Here we just log.

    # 2. Weighted Voting & Confidence
    total_weight = sum(NODES[node_id]['weight'] for node_id in node_data)
    weighted_votes = {}
    
    for node_id, data in node_data.items():
        pred_class = data['prediction']
        confidence = data['confidence']
        weight = NODES[node_id]['weight']
        
        # Vote contribution: normalized weight * confidence
        vote_score = (weight / total_weight) * confidence
        weighted_votes[pred_class] = weighted_votes.get(pred_class, 0) + vote_score

    final_class = max(weighted_votes, key=weighted_votes.get)
    final_confidence = weighted_votes[final_class]
    
    print(f"[Aggregator] Collective Decision: Class {final_class} with confidence {final_confidence:.4f}")

    # 3. Validation by Confidence
    validation_needed = False
    if final_confidence < CONFIDENCE_THRESHOLD:
        print(f"[Aggregator] Confidence {final_confidence:.4f} < {CONFIDENCE_THRESHOLD}. RE-TRIGGERING validation...")
        # Simulation: In a real system, we'd request more data. Here we just flag it.
        validation_needed = True

    # 4. MQTT Telemetry (Supervision Phase 6)
    mqtt_host = os.getenv("MQTT_HOST")
    mqtt_token = os.getenv("MQTT_TOKEN")
    if mqtt_host and mqtt_token:
        try:
            import paho.mqtt.client as mqtt
            client = mqtt.Client()
            client.username_pw_set(mqtt_token)
            client.connect(mqtt_host, 1883, 60)
            
            payload = {
                "collective_class": int(final_class),
                "collective_confidence": float(final_confidence),
                "consensus_achieved": len(set(n['prediction'] for n in node_data.values())) == 1,
                "revalidation_triggered": validation_needed,
                "specimen_id": specimen_id
            }
            client.publish("v1/devices/me/telemetry", json.dumps(payload))
            client.disconnect()
            print(f"[Aggregator] MQTT telemetry sent successfully")
        except Exception as e:
            print(f"[Aggregator] MQTT Error: {e}")

    return {
        "specimen_id": specimen_id,
        "final_class": int(final_class),
        "confidence": float(final_confidence),
        "validation_needed": validation_needed,
        "nodes": node_data
    }

def main():
    if not os.path.exists(SHARED_DIR):
        os.makedirs(SHARED_DIR)
        
    results = []
    # We will evaluate 10 specimens as requested
    for i in range(10):
        res = solve_collective(i)
        if res:
            results.append(res)
            
    output_file = os.path.join(SHARED_DIR, "collective_report.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    # Summarize consensus
    if results:
        consensus_count = 0
        for r in results:
            unique_preds = set(n['prediction'] for n in r['nodes'].values())
            if len(unique_preds) == 1:
                consensus_count += 1
        
        print(f"\n[Aggregator] FINAL EVALUATION SUMMARY:")
        print(f"Total Examples: {len(results)}")
        print(f"Consensus Rate: {consensus_count/len(results)*100:.1f}%")
        print(f"Validation Re-triggers: {sum(1 for r in results if r['validation_needed'])}")

if __name__ == "__main__":
    main()
