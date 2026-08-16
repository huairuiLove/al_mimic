#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Match the AutoDL cgroup quota (cpu.max = 18). Do not set these to 144.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-18}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-18}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-18}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-18}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-18}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset CUDA_DEVICE_MAX_CONNECTIONS || true

CONFIG="${CONFIG:-configs/experiments/mimic_iii/comal.yaml}"
ACTION="${1:-all}"

python -m al_mimic.cli hardware --task mimic_iii --config "$CONFIG"

case "$ACTION" in
  prepare)
    python -m al_mimic.cli prepare --task mimic_iii --config "$CONFIG"
    python -m al_mimic.cli validate-data --task mimic_iii --config "$CONFIG"
    python -m al_mimic.cli explore --task mimic_iii --config "$CONFIG"
    ;;
  active)
    "$ROOT/scripts/experiments/run_four_methods.sh" active
    ;;
  all)
    "$0" prepare
    "$0" active
    ;;
  *)
    echo "usage: $0 {prepare|active|all}" >&2
    exit 2
    ;;
esac
