import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

SEED = 42
np.random.seed(SEED)

DB_PATH = '/Users/ayoub/work/MS-DS_ML_Projects/IOT/mitbih_database'
OUT_DIR = '/Users/ayoub/work/MS-DS_ML_Projects/IOT/collective-intelligence/results/baseline'
os.makedirs(OUT_DIR, exist_ok=True)

AAMI_MAP = {'N':0,'L':0,'R':0,'e':0,'j':0,'A':1,'a':1,'J':1,'S':1,'V':2,'E':2,'F':3,'/':4,'f':4,'Q':4}
FS = 360.0
WINDOW = 180
HALF = WINDOW // 2

def bandpass(x, low=0.5, high=40.0, fs=FS, order=1):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, x)

def minmax(x):
    x = x.astype(np.float32)
    mn, mx = x.min(), x.max()
    return np.zeros_like(x) if mx <= mn else (x - mn) / (mx - mn)

def extract(limit_records=None):
    records = sorted([f[:-4] for f in os.listdir(DB_PATH) if f.endswith('.csv')])
    if limit_records:
        records = records[:limit_records]
    X, y = [], []
    for rid in records:
        sig = pd.read_csv(os.path.join(DB_PATH, f'{rid}.csv'))
        cols = [c for c in sig.columns if np.issubdtype(sig[c].dtype, np.number)]
        if not cols:
            continue
        x = bandpass(sig[cols[0]].values.astype(np.float32))
        ann = pd.read_csv(os.path.join(DB_PATH, f'{rid}annotations.txt'), sep=r'\s+', skiprows=1, header=None, engine='python', on_bad_lines='skip')
        for r, s in zip(ann.iloc[:,1].astype(int).values, ann.iloc[:,2].astype(str).values):
            if s not in AAMI_MAP:
                continue
            st, en = int(r)-HALF, int(r)+HALF
            if st < 0 or en > len(x) or (en-st) != WINDOW:
                continue
            X.append(minmax(x[st:en]))
            y.append(AAMI_MAP[s])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

if __name__ == '__main__':
    X, y = extract()
    np.save(os.path.join(OUT_DIR, 'X_preprocessed.npy'), X)
    np.save(os.path.join(OUT_DIR, 'y_preprocessed.npy'), y)
    print('Saved:', X.shape, y.shape)
