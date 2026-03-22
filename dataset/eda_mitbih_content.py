# SECTION 1.1: BASIC INFO
# =======================

# MIT-BIH Arrhythmia Database
print("=" * 70)
print("SECTION 1.1 — BASIC INFO")
print("=" * 70)

dataset_basic_info = {
    'Name': 'MIT-BIH Arrhythmia Database',
    'Source': 'PhysioNet (https://physionet.org/content/mitdb/1.0.0/)',
    'Format': 'WFDB + CSV (pre-extracted)',
    'Modality': 'Single-lead ECG signals',
    'Download': 'Freely available on PhysioNet',
    'License': 'Open Data Commons Public Domain Dedication & License'
}

for key, value in dataset_basic_info.items():
    print(f"{key:15s}: {value}")

# Count records
csv_files = sorted([f for f in os.listdir(DB_PATH) if f.endswith('.csv')])
record_ids = [os.path.splitext(f)[0] for f in csv_files]
print(f"\nTotal CSV records: {len(record_ids)}")
print(f"Example record IDs: {record_ids[:10]}")

