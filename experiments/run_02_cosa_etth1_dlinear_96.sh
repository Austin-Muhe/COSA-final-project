#!/usr/bin/env bash
set -e

source /home/wwww/miniconda3/etc/profile.d/conda.sh
conda activate cosa

PROJECT_ROOT="/home/wwww/projects/Anhe/COSA-final-project"
COSA_DIR="$PROJECT_ROOT/external/COSA_ICLR2026"

cd "$COSA_DIR"

mkdir -p results/SIMPLE
mkdir -p results/summary/SIMPLE/DLinear/ETTh1

echo "Running minimal COSA test-time adaptation:"
echo "Dataset: ETTh1"
echo "Model: DLinear"
echo "Prediction length: 96"

python main.py \
  DATA.NAME ETTh1 \
  DATA.PRED_LEN 96 \
  MODEL.NAME DLinear \
  MODEL.pred_len 96 \
  TRAIN.ENABLE False \
  TRAIN.CHECKPOINT_DIR ./checkpoints/DLinear/ETTh1_96/ \
  TTA.ENABLE True \
  TTA.SOLVER.BASE_LR 0.001 \
  TTA.SOLVER.WEIGHT_DECAY 0.0001 \
  TTA.SIMPLE.BATCH_SIZE 48 \
  TTA.SIMPLE.STEPS 3 \
  TTA.SIMPLE.BUFFER_CONTEXT_SIZE 10 \
  TTA.SIMPLE.FAST_ADAPTATION True \
  TTA.SIMPLE.PER_BATCH_LR_RESET True \
  TTA.SIMPLE.ADAPTIVE_LR True \
  TTA.SIMPLE.PAAS True \
  TTA.SIMPLE.PERIOD_N 1 \
  RESULT_DIR ./results/SIMPLE/ \
  > ./results/summary/SIMPLE/DLinear/ETTh1/96.txt

echo "COSA result saved to:"
echo "./results/summary/SIMPLE/DLinear/ETTh1/96.txt"
