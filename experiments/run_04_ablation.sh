#!/usr/bin/env bash
set -euo pipefail

source /home/wwww/miniconda3/etc/profile.d/conda.sh
conda activate cosa

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANTS=("original" "vec_gate" "rich_ctx" "ctx_std_only" "cosa_plus")
PRED_LENS=(96 192 336 720)

mkdir -p "$PROJECT_ROOT/results/ablation"
cd "$PROJECT_ROOT"

for VARIANT in "${VARIANTS[@]}"; do
  for PRED_LEN in "${PRED_LENS[@]}"; do
    echo "=== variant=${VARIANT} horizon=${PRED_LEN} ==="
    python experiments/run_cosa_plus.py \
      --dataset ETTh1 \
      --model DLinear \
      --pred_len "${PRED_LEN}" \
      --variant "${VARIANT}" \
      --output_dir ./results/ablation/ \
      > "./results/ablation/${VARIANT}_ETTh1_DLinear_${PRED_LEN}.txt" 2>&1
  done
done
