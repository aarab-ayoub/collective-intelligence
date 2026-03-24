import subprocess
import os
import time

# Champions from Phase 4
CHAMPIONS = {
    "vm1": {"path": "/app/results/optimization/Q2_model.pt", "id": "Q2"},
    "vm2": {"path": "/app/results/optimization/P2_model.pt", "id": "P2"},
    "vm3": {"path": "/app/results/optimization/P3_model.pt", "id": "P3"},
}

def run_cmd(cmd):
    print(f"Exec: {cmd}")
    return subprocess.run(cmd, shell=True)

def main():
    print(">>> Starting Phase 5: Collective Intelligence Hub...")
    
    # Cleanup old results
    run_cmd("rm -rf results/phase5/*.json")
    os.makedirs("results/phase5", exist_ok=True)
    
    # 1. Final Build
    run_cmd("docker compose -f deployment/docker-compose.yml build aggregator")
    
    # 2. Run the 3 champion nodes in COLLECTIVE mode
    # We use 'run' instead of 'up' to control environment variables easily
    processes = []
    for vm_id, info in CHAMPIONS.items():
        cmd = (
            f"docker compose -f deployment/docker-compose.yml run -d "
            f"-e MODEL_PATH={info['path']} "
            f"-e TECH_ID={info['id']} "
            f"-e COLLECTIVE_MODE=true "
            f"-e NUM_SAMPLES=10 "
            f"-e MQTT_HOST=thingsboard "
            f"-e MQTT_TOKEN={vm_id.upper()}_TOKEN "
            f"{vm_id}"
        )
        print(f"Starting {vm_id} with {info['id']}...")
        run_cmd(cmd)

    # 3. Start the Aggregator
    print("Starting Aggregator...")
    run_cmd("docker compose -f deployment/docker-compose.yml run aggregator")

    print("\nPhase 5 Evaluation Complete. Check results/phase5/collective_report.json")

if __name__ == "__main__":
    main()
