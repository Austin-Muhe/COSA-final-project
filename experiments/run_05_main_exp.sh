#!/usr/bin/env bash
set -euo pipefail

source /home/wwww/miniconda3/etc/profile.d/conda.sh
conda activate cosa

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASETS=("ETTh1" "weather")
MODELS=("DLinear" "iTransformer")
VARIANTS=("original" "cosa_plus")
PRED_LENS=(96 192 336 720)

mkdir -p "$PROJECT_ROOT/results/main_exp"
cd "$PROJECT_ROOT"

for DATASET in "${DATASETS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    for VARIANT in "${VARIANTS[@]}"; do
      for PRED_LEN in "${PRED_LENS[@]}"; do
        echo "=== dataset=${DATASET} model=${MODEL} variant=${VARIANT} horizon=${PRED_LEN} ==="
        python experiments/run_cosa_plus.py \
          --dataset "${DATASET}" \
          --model "${MODEL}" \
          --pred_len "${PRED_LEN}" \
          --variant "${VARIANT}" \
          --output_dir ./results/main_exp/ \
          > "./results/main_exp/${VARIANT}_${DATASET}_${MODEL}_${PRED_LEN}.txt" 2>&1
      done
    done
  done
done
