# Baseline Model (Phase 1)

This directory contains the code to train the initial high-precision model before any optimization.

## Data Preparation
Run the script to download and preprocess the MIT-BIH dataset:
```bash
python baseline/prepare_data.py
```

## Training
To train the baseline `ECGNet1D`:
```bash
python baseline/train_baseline.py --epochs 20
```

## Architecture
`ECGNet1D` is a 1D Convolutional Neural Network designed for heartbeat classification. The weights are saved to `results/baseline/baseline_model.pt`.
