#!/usr/bin/env bash
set -euo pipefail

echo "=== Host Resources ==="
uname -a

if command -v sysctl >/dev/null 2>&1; then
  echo "CPU cores: $(sysctl -n hw.ncpu)"
  echo "Memory bytes: $(sysctl -n hw.memsize)"
fi

echo "Disk usage:"
df -h .

echo "Python:"
python --version || true
