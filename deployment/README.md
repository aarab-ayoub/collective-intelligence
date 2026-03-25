# Deployment & Collective Intelligence

This directory contains the infrastructure and orchestrators for Phases 3, 5, and 6.

## Structure
- `docker-compose.yml`: Orchestrates the 3-node cluster with specific CPU/RAM constraints.
- `Dockerfile`: Unified image for all IoT nodes (includes PyTorch, psutil, paho-mqtt).
- `node_eval.py`: The evaluation engine. Supports both individual metrics (Phase 3) and collective inference (Phase 5).
- `aggregator.py`: The collective intelligence hub. Handles weighted voting and confidence re-triggers.

## Usage

### Run Collective Evaluation
To execute the full multi-node voting system (10 specimens):
```bash
python run_phase5.py
```

### Manual Node Run
To run a single node with a specific model:
```bash
docker compose run -e MODEL_PATH=/app/results/optimization/Q2_model.pt -e TECH_ID=Q2 vm1
```

## Resource Constraints
- **VM1**: 1 CPU, 500MB RAM (Micro tier)
- **VM2**: 2 CPU, 1GB RAM (Medium tier)
- **VM3**: 2 CPU, 2GB RAM (Pro tier)
