#!/usr/bin/env bash
set -euo pipefail

source /home/wwww/miniconda3/etc/profile.d/conda.sh
conda activate cosa

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

VARIANTS=("no_tta" "original" "cosa_plus")
SHIFT_TYPES=("level" "variance" "trend" "spike")
MAGNITUDES=("1.0" "2.0" "3.0")
PRED_LEN=720
OUTPUT_DIR="./results/blackswan_etth1_magnitude_sweep_h720/"
RUN_LOG="${OUTPUT_DIR}/run_log.txt"

mkdir -p "$OUTPUT_DIR"
touch "$RUN_LOG"

for MAGNITUDE in "${MAGNITUDES[@]}"; do
  for VARIANT in "${VARIANTS[@]}"; do
    for SHIFT_TYPE in "${SHIFT_TYPES[@]}"; do
      if [[ "$MAGNITUDE" == "3.0" ]]; then
        REUSED_JSON="./results/blackswan_mag3_720/${VARIANT}_${SHIFT_TYPE}_m3_${PRED_LEN}.json"
        if [[ -f "$REUSED_JSON" ]]; then
          echo "=== reuse existing ${REUSED_JSON} ===" | tee -a "$RUN_LOG"
          continue
        fi
      fi
      RESULT_JSON="${OUTPUT_DIR}/${VARIANT}_${SHIFT_TYPE}_m${MAGNITUDE%.*}_${PRED_LEN}.json"
      if [[ -f "$RESULT_JSON" ]]; then
        echo "=== skip existing ${RESULT_JSON} ===" | tee -a "$RUN_LOG"
        continue
      fi
      echo "=== ETTh1 DLinear h720 mag=${MAGNITUDE} variant=${VARIANT} shift=${SHIFT_TYPE} ===" | tee -a "$RUN_LOG"
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
done

python experiments/blackswan/make_small_artifacts.py \
  --input_dir "${OUTPUT_DIR}" \
  --input_dir ./results/blackswan_mag3_720/ \
  --output_dir ./results/final_figures/ \
  --experiment magnitude

echo "Done. Results in ${OUTPUT_DIR} and results/final_figures/"
