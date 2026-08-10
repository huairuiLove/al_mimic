#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ACTION="${1:-active}"
CONFIGS=(
  configs/mimic_comal.yaml
  configs/mimic_mm_comal.yaml
  configs/mimic_modis.yaml
  configs/mimic_mosaic.yaml
)

for config in "${CONFIGS[@]}"; do
  python main.py "$ACTION" --config "$config"
done

if [[ "$ACTION" == "active" ]]; then
  for config in "${CONFIGS[@]}"; do
    python main.py visualize --config "$config"
  done
  python scripts/compare_experiments.py \
    --experiment experiments/mimic_iii_yang_wu_bertencoder_comal CoMAL \
    --experiment experiments/mimic_iii_yang_wu_bertencoder_mm_comal MM-CoMAL \
    --experiment experiments/mimic_iii_yang_wu_bertencoder_modis MoDIS \
    --experiment experiments/mimic_iii_yang_wu_bertencoder_mosaic MoSAIC \
    --output experiments/yang_wu_four_methods_recall_at_30.png
fi
