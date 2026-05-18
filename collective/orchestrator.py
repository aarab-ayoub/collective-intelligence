import subprocess
import os
import time
from pathlib import Path

# Champions from Phase 4
CHAMPIONS = {
    "vm1": {"path": "/app/results/optimization/Q2_model.pt", "id": "Q2"},
    "vm2": {"path": "/app/results/optimization/P2_model.pt", "id": "P2"},
    "vm3": {"path": "/app/results/optimization/P3_model.pt", "id": "P3"},
}

# MQTT Tokens from thingsboard
MQTT_TOKENS = {
    "vm1": "24I0bz103fw2rt8X6W7z",
    "vm2": "miKQqtXvO6x7eysL3w4L",
    "vm3": "lgDfgBFuSKR7679PBhIo",
    "aggregator": "6XIKx1kXMSsCIFkXWJwg"
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPOSE_FILE = PROJECT_ROOT / "environment" / "docker-compose.yml"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase5"

def run_cmd(cmd):
    print(f"Exec: {cmd}")
    return subprocess.run(cmd, shell=True)

def main():
    print(">>> Starting Phase 5: Collective Intelligence Hub...")
    
    # Cleanup old results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for json_file in RESULTS_DIR.glob("*.json"):
        json_file.unlink()
    
    # 1. Final Build
    run_cmd(f"docker compose -f {COMPOSE_FILE} build aggregator")
    
    # 2. Run the 3 champion nodes in COLLECTIVE mode
    # We use 'run' instead of 'up' to control environment variables easily
    for vm_id, info in CHAMPIONS.items():
        cmd = (
            f"docker compose -f {COMPOSE_FILE} run --rm -d "
            f"-e MODEL_PATH={info['path']} "
            f"-e TECH_ID={info['id']} "
            f"-e SHARED_DIR=/app/results/phase5 "
            f"-e COLLECTIVE_MODE=true "
            f"-e NUM_SAMPLES=10 "
            f"-e SEND_NODE_TELEMETRY=true "
            f"-e MQTT_HOST=thingsboard "
            f"-e MQTT_TOKEN={MQTT_TOKENS[vm_id]} "
            f"{vm_id}"
        )
        print(f"Starting {vm_id} with {info['id']}...")
        run_cmd(cmd)

    # Give nodes a short moment to initialize and write their first outputs
    time.sleep(2)

    # 3. Start the Aggregator
    print("Starting Aggregator...")
    agg_cmd = (
        f"docker compose -f {COMPOSE_FILE} run --rm "
        f"-e SHARED_DIR=/app/results/phase5 "
        f"-e MQTT_HOST=thingsboard "
        f"-e MQTT_TOKEN={MQTT_TOKENS['aggregator']} "
        f"aggregator"
    )
    run_cmd(agg_cmd)

    print(f"\nPhase 5 Evaluation Complete. Check {RESULTS_DIR / 'collective_report.json'}")

if __name__ == "__main__":
    main()
