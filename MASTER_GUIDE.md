# IoT ECG Project: Master Guide (Phase 1-6)

This guide provides the complete sequence of commands for the entire life cycle of the Optimized ECG IoT system.

---

## 🚀 Overview of Phases
1. **Phase 1-2**: Baseline Training & Optimization (8 Techniques)
2. **Phase 3**: Performance Matrix (Deployment on 3 VM Tiers)
3. **Phase 4**: Champion Selection (Weighted Scoring)
4. **Phase 5-6**: Collective Intelligence & ThingsBoard Supervision

---

## 🛠 Prerequisites
- **Python**: 3.13.9
- **Docker**: Desktop for Mac (for VM simulation)
- **Shared Volumes**: The system uses `/app/results` and `/mitbih` for data persistence.

---

## 📖 Phase 1: Baseline Model
Train the initial high-precision ECGNet1D model.
```bash
# 1. Prepare Dataset
python baseline/prepare_data.py

# 2. Train Baseline
python baseline/train.py --epochs 20
```

---

## ⚡️ Phase 2: Model Optimization
Generate 8 optimized variants (Quantization Q1-Q5 & Pruning P1-P3).
```bash
# Run all optimization techniques and generate .pt files
python optimization/compare_techniques.py
```

---

## 📊 Phase 3 & 4: Deployment & Selection
Evaluate optimization performance across hardware tiers and select Champions.
```bash
# 1. Build Docker Environment
docker compose -f environment/docker-compose.yml build

# 2. Calculate Weighted Scores (Champion Selection)
python optimization/select_best.py
```

---

## 🧠 Phase 5 & 6: Collective Intelligence & Supervision
Run the multi-node voting hub with real-time ThingsBoard telemetry.

### 1. Initialize Supervision (ThingsBoard)
```bash
# Start ThingsBoard and MQTT Broker
docker compose -f environment/docker-compose.yml up -d thingsboard
```
*Wait for UI at http://localhost:8080. Access with `sysadmin@thingsboard.org` / `sysadmin`.*

### 2. Run Collective Intelligence Hub
This command launches 3 VM containers (VM1-VM3) and the Aggregator.
```bash
python collective/orchestrator.py
```

---

## 📁 Project Structure Summary
- `baseline/`: Training scripts and base weights.
- `optimization/`: Implementation of 8 optimization techniques.
- `environment/`: Docker Compose and resource limit configurations.
- `collective/`: Orchestrator and Consensus Aggregator.
- `deployment/`: Node evaluation engine for edge inference.
- `thingsboard/`: Dashboard JSON for cloud monitoring.
- `results/`: CSV reports and performance metrics.
