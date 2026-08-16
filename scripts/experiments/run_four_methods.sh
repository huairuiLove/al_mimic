#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ACTION="${1:-active}"
for method in comal mm_comal modis mosaic; do
  config="configs/experiments/mimic_iii/${method}.yaml"
  if [[ "$ACTION" == "active" ]]; then
    python -m al_mimic.cli active --task mimic_iii --method "$method" --config "$config"
  else
    python -m al_mimic.cli "$ACTION" --task mimic_iii --config "$config"
  fi
done

if [[ "$ACTION" == "active" ]]; then
  for method in comal mm_comal modis mosaic; do
    python -m al_mimic.cli visualize \
      --task mimic_iii \
      --config "configs/experiments/mimic_iii/${method}.yaml"
  done
  python scripts/evaluation/compare_experiments.py \
    --experiment experiments/mimic_iii_yang_wu_bertencoder_comal CoMAL \
    --experiment experiments/mimic_iii_yang_wu_bertencoder_mm_comal MM-CoMAL \
    --experiment experiments/mimic_iii_yang_wu_bertencoder_modis MoDIS \
    --experiment experiments/mimic_iii_yang_wu_bertencoder_mosaic MoSAIC \
    --output experiments/yang_wu_four_methods_recall_at_30.png
fi
