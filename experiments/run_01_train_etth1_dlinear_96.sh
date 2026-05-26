#!/usr/bin/env bash
set -e

source /home/wwww/miniconda3/etc/profile.d/conda.sh
conda activate cosa

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COSA_DIR="$PROJECT_ROOT/external/COSA_ICLR2026"

cd "$COSA_DIR"

mkdir -p checkpoints/DLinear/ETTh1_96
mkdir -p results/logs

echo "Running minimal base-model training:"
echo "Dataset: ETTh1"
echo "Model: DLinear"
echo "Prediction length: 96"

python main.py \
  DATA.NAME ETTh1 \
  DATA.PRED_LEN 96 \
  MODEL.NAME DLinear \
  MODEL.pred_len 96 \
  TRAIN.ENABLE True \
  TRAIN.CHECKPOINT_DIR ./checkpoints/DLinear/ETTh1_96/
