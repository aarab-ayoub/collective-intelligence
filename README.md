# Collective Intelligence — PTB-XL Project Starter

## Project structure

- `dataset/raw/` → raw datasets (PTB-XL)
- `dataset/processed/` → processed arrays and metadata generated in Phase 0
- `notebooks/phase0/` → EDA + preprocessing notebook
- `phase1/models/` → Phase 1 Python training scripts
- `results/phase0/` → figures/tables from EDA
- `results/phase1/` → model checkpoints and metrics

## Dataset choice

You chose **PTB-XL**, which is a strong choice for ECG classification.

Expected raw dataset path for this project:
- `collective-intelligence/dataset/raw/ptb-xl/`

If your dataset currently exists at `IOT/ptb-xl/`, either:
1. copy it into `collective-intelligence/dataset/raw/ptb-xl/`, or
2. create a symlink named `ptb-xl` inside `collective-intelligence/dataset/raw/`.

## Workflow

1. Open `notebooks/phase0/phase0_eda_preprocessing.ipynb`
2. Run all cells to generate processed files in `dataset/processed/`
3. Train baseline model:

```bash
python phase1/models/train_phase1_cnn.py --data-dir dataset/processed --out-dir results/phase1 --epochs 10
```

## Final baseline

The locked final Phase 1 baseline metrics and configuration are documented in `RESULTS.md`.

### Reproduce final Phase 1 run

```bash
python phase1/models/train_phase1_cnn.py \
	--data-dir dataset/processed \
	--out-dir results/phase1 \
	--seed 42 \
	--epochs 35 \
	--batch-size 64 \
	--lr 9e-4 \
	--dropout 0.28 \
	--base-channels 28 \
	--weight-decay 6e-4 \
	--label-smoothing 0.03 \
	--early-stopping-patience 10 \
	--lr-patience 4 \
	--lr-factor 0.5 \
	--min-lr 1e-5 \
	--min-delta 0.0008
```

### Evaluate on test set only

```bash
python phase1/models/train_phase1_cnn.py \
	--data-dir dataset/processed \
	--batch-size 64 \
	--dropout 0.28 \
	--base-channels 28 \
	--label-smoothing 0.03 \
	--eval-only \
	--checkpoint results/phase1/best_model.pt
```
