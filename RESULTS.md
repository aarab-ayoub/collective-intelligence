# Final Phase 1 Baseline Results

This file locks the current **official Phase 1 baseline** for the PTB-XL project.

## Run artifact location

- `results/phase1/best_model.pt`
- `results/phase1/history.json`
- `results/phase1/summary.json`

## Training setup (final run)

- Device: `mps`
- Epochs requested: `35`
- Epochs run (early stopping): `24`
- Batch size: `64`
- Learning rate: `9e-4`
- Dropout: `0.28`
- Base channels: `28`
- Weight decay: `6e-4`
- Label smoothing: `0.03`
- Early stopping patience: `10`
- LR scheduler: `ReduceLROnPlateau` (`patience=4`, `factor=0.5`, `min_lr=1e-5`)
- Min delta (val loss improvement): `0.0008`

## Dataset split

- Train: `2382`
- Validation: `313`
- Test: `305`
- Classes: `5`

## Key metrics

- Best validation **accuracy** epoch: `22`
  - Val accuracy: `0.7572`
  - Val F1: `0.6556`
  - Val loss: `0.8380`

- Best validation **loss** epoch: `14`
  - Val loss: `0.8177`
  - Val accuracy: `0.7348`
  - Val F1: `0.6083`

- Final epoch (`24`) overfitting check:
  - Train accuracy: `0.8283`
  - Validation accuracy: `0.7125`
  - Accuracy gap: `0.1158`
  - Train loss: `0.5749`
  - Validation loss: `0.8789`
  - Loss gap: `0.3040`

- Test metrics (best checkpoint by validation loss):
  - Test loss: `0.9535`
  - Test accuracy: `0.7016`
  - Test F1: `0.6253`

## Baseline decision

Use `results/phase1/best_model.pt` as the official Phase 1 baseline checkpoint.
