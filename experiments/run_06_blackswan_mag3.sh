#!/usr/bin/env bash
set -euo pipefail

source /home/wwww/miniconda3/etc/profile.d/conda.sh
conda activate cosa

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

VARIANTS=("no_tta" "original" "cosa_plus")
SHIFT_TYPES=("level" "variance" "trend" "spike")
MAGNITUDE=3.0

mkdir -p ./results/blackswan_mag3/
: > ./results/blackswan_mag3/run_log.txt

for VARIANT in "${VARIANTS[@]}"; do
  for SHIFT_TYPE in "${SHIFT_TYPES[@]}"; do
    echo "=== mag3 variant=${VARIANT} shift=${SHIFT_TYPE} ===" | tee -a ./results/blackswan_mag3/run_log.txt
    python experiments/blackswan/run_blackswan.py \
      --dataset ETTh1 \
      --model DLinear \
      --pred_len 96 \
      --variant "${VARIANT}" \
      --shift_type "${SHIFT_TYPE}" \
      --magnitude "${MAGNITUDE}" \
      --output_dir ./results/blackswan_mag3/ \
      >> ./results/blackswan_mag3/run_log.txt 2>&1
  done
done

echo "Done. Results in results/blackswan_mag3/"
