#!/usr/bin/env bash
# Sequential MIMIC-III active-learning sweep over the formal methods.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
unset CUDA_DEVICE_MAX_CONNECTIONS || true

LOG_DIR="${LOG_DIR:-$ROOT/experiments/_logs}"
mkdir -p "$LOG_DIR"

if [[ $# -gt 0 ]]; then
  METHODS=("$@")
else
  METHODS=(random comal modis modimix mosaic)
fi

echo "[sweep] start $(date -Is) methods=${METHODS[*]}" | tee -a "$LOG_DIR/sweep.log"

for method in "${METHODS[@]}"; do
  config="configs/experiments/mimic_iii/${method}.yaml"
  method_log="$LOG_DIR/${method}.log"
  echo "[sweep] $method begin $(date -Is)" | tee -a "$LOG_DIR/sweep.log"
  python -m al_mimic.cli hardware --task mimic_iii --config "$config" | tee -a "$method_log"
  python -m al_mimic.cli active --task mimic_iii --method "$method" --config "$config" 2>&1 | tee -a "$method_log"
  echo "[sweep] $method end $(date -Is)" | tee -a "$LOG_DIR/sweep.log"
done

echo "[sweep] all done $(date -Is)" | tee -a "$LOG_DIR/sweep.log"
