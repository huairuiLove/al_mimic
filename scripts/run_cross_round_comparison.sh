#!/usr/bin/env bash
set -euo pipefail

python main.py active --config configs/mimic_cross_round_cold.yaml
python main.py active --config configs/mimic_cross_round_inherit.yaml
python main.py visualize --config configs/mimic_cross_round_cold.yaml
python main.py visualize --config configs/mimic_cross_round_inherit.yaml
python scripts/compare_experiments.py \
  --experiment experiments/mimic_cross_round_cold Cold-start \
  --experiment experiments/mimic_cross_round_inherit Inherit \
  --output experiments/cross_round_comparison.png
