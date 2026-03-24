# Phase 5: Collective Intelligence (Multi-Node Voting)

This phase implements a collaborative inference system where three distinct IoT nodes (VM1, VM2, VM3) run different optimized models and a centralized **Aggregator** performs majority voting to produce the final classification.

## User Review Required
> [!IMPORTANT]
> I will use the three model variants requested: **Baseline (B0)**, **Structured Pruning (P2)**, and **Weight-only FP16 (Q4)**.
> Node assignments:
> - **Node 1 (VM1 - 500MB):** Baseline (B0)
> - **Node 2 (VM2 - 1GB):** Structured Pruning (P2)
> - **Node 3 (VM3 - 2GB):** Weight-only FP16 (Q4)

## Proposed Changes

### [Deployment Infrastructure]

#### [MODIFY] [docker-compose.yml](file:///Users/ayoub/work/MS-DS_ML_Projects/IOT/collective-intelligence/deployment/docker-compose.yml)
- Add an `aggregator` service.
- Ensure all nodes share a common volume `/app/results/phase5` to exchange predictions.

#### [NEW] [aggregator.py](file:///Users/ayoub/work/MS-DS_ML_Projects/IOT/collective-intelligence/deployment/aggregator.py)
- Implement a script that waits for predictions from all three nodes.
- Performs **Majority Voting**: Final Class = mode([Node1, Node2, Node3]).
- Calculates the "Collective Accuracy" and saves it to a JSON report.

#### [MODIFY] [node_eval.py](file:///Users/ayoub/work/MS-DS_ML_Projects/IOT/collective-intelligence/deployment/node_eval.py)
- Update to accept a flag to save individual specimen predictions to `/app/results/phase5/`.

### [Orchestration]

#### [NEW] [run_phase5.py](file:///Users/ayoub/work/MS-DS_ML_Projects/IOT/collective-intelligence/run_phase5.py)
- Orchestrates the start of the 3-node cluster and the aggregator.
- Feeds a batch of 100 test samples to all nodes simultaneously.
- Collects and displays the final collective performance metrics.

## Verification Plan

### Automated Tests
1. Run `python run_phase5.py`.
2. Verify that `results/phase5/collective_report.json` is generated.
3. Confirm that "Collective Accuracy" is equal to or better than the individual node accuracies on conflicting samples.

### Manual Verification
- Check the `aggregator` logs to see instances where nodes disagreed and how the majority vote resolved the conflict.
