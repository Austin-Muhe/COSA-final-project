#!/usr/bin/env bash
set -euo pipefail

source /home/wwww/miniconda3/etc/profile.d/conda.sh
conda activate cosa

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASETS=("ETTh1")
MODEL="DLinear"
VARIANTS=("original" "cosa_plus")
# Keep this list aligned with available checkpoints under
# external/COSA_ICLR2026/checkpoints/DLinear/.
PRED_LENS=(96)

mkdir -p "$PROJECT_ROOT/results/main_exp_dlinear_first"
cd "$PROJECT_ROOT"

for DATASET in "${DATASETS[@]}"; do
  for VARIANT in "${VARIANTS[@]}"; do
    for PRED_LEN in "${PRED_LENS[@]}"; do
      echo "=== DLinear-first dataset=${DATASET} variant=${VARIANT} horizon=${PRED_LEN} ==="
      python experiments/run_cosa_plus.py \
        --dataset "${DATASET}" \
        --model "${MODEL}" \
        --pred_len "${PRED_LEN}" \
        --variant "${VARIANT}" \
        --output_dir ./results/main_exp_dlinear_first/ \
        > "./results/main_exp_dlinear_first/${VARIANT}_${DATASET}_${MODEL}_${PRED_LEN}.txt" 2>&1
    done
  done
done
