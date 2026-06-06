#!/usr/bin/env bash
set -euo pipefail

source /home/wwww/miniconda3/etc/profile.d/conda.sh
conda activate cosa

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

VARIANTS=("no_tta" "original" "cosa_plus")
SHIFT_TYPES=("level" "variance" "trend" "spike")
MAGNITUDES=(1.0 3.0 5.0)

mkdir -p ./results/blackswan/
: > ./results/blackswan/run_log.txt

for VARIANT in "${VARIANTS[@]}"; do
  for SHIFT_TYPE in "${SHIFT_TYPES[@]}"; do
    for MAG in "${MAGNITUDES[@]}"; do
      echo "=== variant=${VARIANT} shift=${SHIFT_TYPE} mag=${MAG} ===" | tee -a ./results/blackswan/run_log.txt
      python experiments/blackswan/run_blackswan.py \
        --dataset ETTh1 \
        --model DLinear \
        --pred_len 96 \
        --variant "${VARIANT}" \
        --shift_type "${SHIFT_TYPE}" \
        --magnitude "${MAG}" \
        --output_dir ./results/blackswan/ \
        >> ./results/blackswan/run_log.txt 2>&1
    done
  done
done

echo "Done. Results in results/blackswan/"
