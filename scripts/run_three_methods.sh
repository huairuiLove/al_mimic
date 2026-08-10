#!/usr/bin/env bash
set -euo pipefail

phase="${1:-all}"
shared_config="configs/mimic_comal.yaml"
configs=(
  "configs/mimic_comal.yaml"
  "configs/mimic_mm_comal.yaml"
  "configs/mimic_mosaic.yaml"
  "configs/mimic_modis.yaml"
)

case "${phase}" in
  prepare)
    python main.py prepare --config "${shared_config}"
    ;;
  features)
    python main.py features --config "${shared_config}"
    ;;
  active)
    for config in "${configs[@]}"; do
      python main.py active --config "${config}"
    done
    python scripts/compare_experiments.py \
      --experiment experiments/mimic_iii_comal CoMAL \
      --experiment experiments/mimic_iii_mm_comal MM-CoMAL \
      --experiment experiments/mimic_iii_mosaic MoSAIC \
      --experiment experiments/mimic_iii_modis MoDIS \
      --output experiments/four_method_comparison.png
    ;;
  all)
    python main.py prepare --config "${shared_config}"
    python main.py validate-data --config "${shared_config}"
    python main.py features --config "${shared_config}"
    for config in "${configs[@]}"; do
      python main.py active --config "${config}"
      python main.py visualize --config "${config}"
    done
    python scripts/compare_experiments.py \
      --experiment experiments/mimic_iii_comal CoMAL \
      --experiment experiments/mimic_iii_mm_comal MM-CoMAL \
      --experiment experiments/mimic_iii_mosaic MoSAIC \
      --experiment experiments/mimic_iii_modis MoDIS \
      --output experiments/four_method_comparison.png
    ;;
  *)
    echo "usage: $0 [prepare|features|active|all]" >&2
    exit 2
    ;;
esac
