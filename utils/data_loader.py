import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from .config import DATA_DIR, SEED

def get_train_loader(batch_size=256, subset_frac=1.0):
    train_df = pd.read_csv(DATA_DIR / "mitbih_train.csv", header=None)
    if subset_frac < 1.0:
        train_df = train_df.sample(frac=subset_frac, random_state=SEED)
    X = train_df.iloc[:, :-1].values.astype(np.float32)
    y = train_df.iloc[:, -1].values.astype(np.int64)
    x_t = torch.tensor(X).unsqueeze(1)
    y_t = torch.tensor(y)
    return DataLoader(TensorDataset(x_t, y_t), batch_size=batch_size, shuffle=True)

def get_test_loader(batch_size=1024):
    test_df = pd.read_csv(DATA_DIR / "mitbih_test.csv", header=None)
    X_test = test_df.iloc[:, :-1].values.astype(np.float32)
    y_test = test_df.iloc[:, -1].values.astype(np.int64)
    x_test_t = torch.tensor(X_test).unsqueeze(1)
    y_test_t = torch.tensor(y_test)
    return DataLoader(TensorDataset(x_test_t, y_test_t), batch_size=batch_size, shuffle=False)
