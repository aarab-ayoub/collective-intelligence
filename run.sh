#!/bin/bash

# Run baseline training
python baseline/train.py

# Run all optimizations
python optimization/P1_unstructured_pruning/optimize.py
python optimization/P2_structured_pruning/optimize.py
python optimization/P3_magnitude_pruning/optimize.py
python optimization/Q1_dynamic_quant/optimize.py
python optimization/Q2_static_ptq/optimize.py
python optimization/Q3_qat/optimize.py
python optimization/Q4_weight_only/optimize.py
python optimization/Q5_mixed_precision/optimize.py

# Compare all optimizations
python optimization/compare_techniques.py

# Run Phase 3 - VM Deployment
# python phase3/deploy_to_vm.py --vm_name VM1 --memory 500MB --model_path results/optimization/P2_model.pt
# python phase3/deploy_to_vm.py --vm_name VM2 --memory 1GB --model_path results/optimization/P2_model.pt
# python phase3/deploy_to_vm.py --vm_name VM3 --memory 2GB --model_path results/optimization/P2_model.pt

# Run Phase 5 - Collective Intelligence
# python phase5/collective_intelligence.py --aggregator_vm VM1 --worker_vms VM2 VM3

# Run Phase 6 - Supervision
# python phase6/supervision.py --aggregator_vm VM1 --thingsboard_host localhost --thingsboard_port 1883 --thingsboard_token <YOUR_TOKEN>

# echo "All phases completed!"