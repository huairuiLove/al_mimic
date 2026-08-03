#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-all}"
CONFIG="${CONFIG:-configs/mimic_a800_144c.yaml}"

case "$ACTION" in
  prepare)
    python main.py prepare --config "$CONFIG"
    python main.py validate-data --config "$CONFIG"
    python main.py explore --config "$CONFIG"
    ;;
  features)
    python main.py features --config "$CONFIG"
    ;;
  active)
    python main.py active --config "$CONFIG"
    python main.py active --config configs/mimic_a800_random.yaml
    python main.py visualize --config "$CONFIG"
    python main.py visualize --config configs/mimic_a800_random.yaml
    python scripts/compare_experiments.py \
      --experiment experiments/mimic_iii_comal_a800 CoMAL \
      --experiment experiments/mimic_iii_random_a800 Random \
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
