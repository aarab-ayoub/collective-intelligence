# Model Optimization (Phase 2)

This directory implements 8 different optimization techniques for the `ECGNet1D` model.

## Techniques
- **Q1**: Dynamic Quantization
- **Q2**: Static Post-Training Quantization (Champion for VM1)
- **Q3**: Quantization-Aware Training (QAT)
- **Q4**: Weight-Only Quantization (Int8)
- **Q5**: Mixed Precision
- **P1**: Unstructured Pruning (50%)
- **P2**: Structured Pruning (30%) (Champion for VM2)
- **P3**: Global Magnitude Pruning (40%) (Champion for VM3)

## Scripts
- `optimize.py`: Main entry point for generating all 8 versions.
- `select_best.py`: Phase 4 script for calculating weighted scores per VM tier.

## Results
Summarized table can be found in `results/phase3/phase3_full_matrix.csv`.
Champion justifications are available in the project walkthrough.
