#!/usr/bin/env bash
set -e

source /home/wwww/miniconda3/etc/profile.d/conda.sh
conda activate cosa

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COSA_DIR="$PROJECT_ROOT/external/COSA_ICLR2026"

cd "$COSA_DIR"

echo "Current directory:"
pwd

echo ""
echo "Python environment:"
which python
python --version

echo ""
echo "Checking Python packages and GPU:"
python - <<'PY'
import torch
import numpy
import pandas
import sklearn
import scipy
import matplotlib

print("basic packages ok")
print("torch:", torch.__version__)
print("numpy:", numpy.__version__)
print("scipy:", scipy.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
PY

echo ""
echo "Checking ETTh1 dataset:"
test -f datasets/ETTh1.csv

python - <<'PY'
import pandas as pd

path = "datasets/ETTh1.csv"
df = pd.read_csv(path)

print("Loaded:", path)
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print(df.head())
PY

echo ""
echo "Environment and dataset check completed."
