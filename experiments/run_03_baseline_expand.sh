#!/usr/bin/env bash
set -euo pipefail

source /home/wwww/miniconda3/etc/profile.d/conda.sh
conda activate cosa

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COSA_DIR="$PROJECT_ROOT/external/COSA_ICLR2026"
PRED_LENS=(96 192 336 720)

cd "$COSA_DIR"
mkdir -p "$PROJECT_ROOT/results/baseline"

for PRED_LEN in "${PRED_LENS[@]}"; do
  echo "=== Running COSA-F dataset=ETTh1 model=DLinear horizon=${PRED_LEN} ==="
  python main.py \
    DATA.NAME ETTh1 \
    DATA.PRED_LEN "${PRED_LEN}" \
    MODEL.NAME DLinear \
    MODEL.pred_len "${PRED_LEN}" \
    TRAIN.ENABLE False \
    TRAIN.CHECKPOINT_DIR "./checkpoints/DLinear/ETTh1_${PRED_LEN}/" \
    TTA.ENABLE True \
    TTA.SOLVER.BASE_LR 0.001 \
    TTA.SOLVER.WEIGHT_DECAY 0.0001 \
    TTA.COSA.BATCH_SIZE 48 \
    TTA.COSA.STEPS 3 \
    TTA.COSA.BUFFER_CONTEXT_SIZE 10 \
    TTA.COSA.FAST_ADAPTATION True \
    TTA.COSA.ADAPTIVE_LR True \
    TTA.COSA.PER_BATCH_LR_RESET True \
    TTA.COSA.PAAS False \
    RESULT_DIR "./results/SIMPLE/" \
    > "$PROJECT_ROOT/results/baseline/DLinear_ETTh1_${PRED_LEN}.txt" 2>&1
done

echo "Done. Check $PROJECT_ROOT/results/baseline/"
