#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-all}"

case "$ACTION" in
  prepare)
    python main.py prepare --config configs/mimic_comal.yaml
    python main.py validate-data --config configs/mimic_comal.yaml
    python main.py explore --config configs/mimic_comal.yaml
    ;;
  features)
    python main.py features --config configs/mimic_comal.yaml
    ;;
  active)
    python main.py active --config configs/mimic_comal.yaml
    python main.py active --config configs/mimic_random.yaml
    python main.py visualize --config configs/mimic_comal.yaml
    python main.py visualize --config configs/mimic_random.yaml
    python scripts/compare_experiments.py \
      --experiment experiments/mimic_iii_comal CoMAL \
      --experiment experiments/mimic_iii_random Random \
      --output experiments/comal_vs_random.png
    ;;
  all)
    "$0" prepare
    "$0" features
    "$0" active
    ;;
  *)
    echo "usage: $0 {prepare|features|active|all}" >&2
    exit 2
    ;;
esac

