#!/usr/bin/env bash
set -euo pipefail

source /home/wwww/miniconda3/etc/profile.d/conda.sh
conda activate cosa

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

VARIANTS=("vec_gate" "rich_ctx")
SHIFT_TYPES=("level" "variance" "trend" "spike")
MAGNITUDE=3.0
PRED_LEN=720
OUTPUT_DIR="./results/blackswan_etth1_ablation_h720_mag3_missing/"
RUN_LOG="${OUTPUT_DIR}/run_log.txt"

mkdir -p "$OUTPUT_DIR"
touch "$RUN_LOG"

for VARIANT in "${VARIANTS[@]}"; do
  for SHIFT_TYPE in "${SHIFT_TYPES[@]}"; do
    RESULT_JSON="${OUTPUT_DIR}/${VARIANT}_${SHIFT_TYPE}_m3_${PRED_LEN}.json"
    if [[ -f "$RESULT_JSON" ]]; then
      echo "=== skip existing ${RESULT_JSON} ===" | tee -a "$RUN_LOG"
      continue
    fi
    echo "=== ETTh1 DLinear h720 mag3 variant=${VARIANT} shift=${SHIFT_TYPE} ===" | tee -a "$RUN_LOG"
    python experiments/blackswan/run_blackswan.py \
      --dataset ETTh1 \
      --model DLinear \
      --pred_len "${PRED_LEN}" \
      --variant "${VARIANT}" \
      --shift_type "${SHIFT_TYPE}" \
      --magnitude "${MAGNITUDE}" \
      --output_dir "${OUTPUT_DIR}" \
      >> "$RUN_LOG" 2>&1
  done
done

python experiments/blackswan/make_small_artifacts.py \
  --input_dir ./results/blackswan_mag3_720/ \
  --input_dir "${OUTPUT_DIR}" \
  --output_dir ./results/final_figures/ \
  --experiment ablation

echo "Done. Results in ${OUTPUT_DIR} and results/final_figures/"
