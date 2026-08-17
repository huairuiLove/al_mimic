#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ACTION="${1:-all}"
CONFIG="${CONFIG:-configs/experiments/mimic_iii/comal.yaml}"

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
