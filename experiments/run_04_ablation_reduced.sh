#!/usr/bin/env bash
set -euo pipefail

source /home/wwww/miniconda3/etc/profile.d/conda.sh
conda activate cosa

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANTS=("original" "vec_gate" "rich_ctx" "ctx_std_only" "cosa_plus")
# Keep this list aligned with available checkpoints under
# external/COSA_ICLR2026/checkpoints/DLinear/. Add 720 after placing
# checkpoints/DLinear/ETTh1_720/checkpoint_best.pth.
PRED_LENS=(96)

mkdir -p "$PROJECT_ROOT/results/ablation_reduced"
cd "$PROJECT_ROOT"

for VARIANT in "${VARIANTS[@]}"; do
  for PRED_LEN in "${PRED_LENS[@]}"; do
    echo "=== reduced ablation variant=${VARIANT} horizon=${PRED_LEN} ==="
    python experiments/run_cosa_plus.py \
      --dataset ETTh1 \
      --model DLinear \
      --pred_len "${PRED_LEN}" \
      --variant "${VARIANT}" \
      --output_dir ./results/ablation_reduced/ \
      > "./results/ablation_reduced/${VARIANT}_ETTh1_DLinear_${PRED_LEN}.txt" 2>&1
  done
done
