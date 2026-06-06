#!/usr/bin/env bash
set -e

source /home/wwww/miniconda3/etc/profile.d/conda.sh
conda activate cosa

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python test_black_swan/run_exchange_black_swan_test.py \
  --model DLinear \
  --pred-len 96 \
  --epochs 30 \
  --severity 0 5 10
