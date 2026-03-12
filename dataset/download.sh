#!/usr/bin/env bash
# download.sh — Downloads the PTB-XL ECG dataset from PhysioNet.
#
# PTB-XL: 21,837 clinical 12-lead ECGs, 10-second recordings at 500 Hz
# Source : https://physionet.org/content/ptb-xl/1.0.3/
# Size   : ~2.5 GB
#
# Requirements:
#   pip install wfdb requests
#   OR: wget + unzip (method 2 below)

set -euo pipefail

OUTPUT_DIR="./raw/ptb-xl"
PHYSIONET_URL="https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"

mkdir -p "$OUTPUT_DIR"

# ── Method 1: Python wfdb downloader (recommended) ─────────────────────────
if command -v python3 &>/dev/null; then
  echo "[INFO] Downloading PTB-XL via wfdb..."
  python3 - <<'EOF'
import wfdb
wfdb.dl_database('ptb-xl', dl_dir='./raw/ptb-xl')
print('[INFO] Download complete via wfdb.')
EOF

# ── Method 2: wget fallback ─────────────────────────────────────────────────
elif command -v wget &>/dev/null; then
  echo "[INFO] Downloading PTB-XL via wget (this may take a while)..."
  wget -c "$PHYSIONET_URL" -O /tmp/ptbxl.zip
  unzip -q /tmp/ptbxl.zip -d "$OUTPUT_DIR"
  echo "[INFO] Download complete."
else
  echo "[ERROR] Neither python3+wfdb nor wget found. Install one and retry."
  exit 1
fi

echo "[INFO] PTB-XL files saved to $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
