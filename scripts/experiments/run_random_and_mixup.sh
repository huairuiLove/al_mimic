#!/usr/bin/env bash
# Serial queue for the acquisition control and the two mixup arms.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ACTION="${1:-active}"
METHODS=(random random comal)
CONFIGS=(
  configs/experiments/mimic_iii/random.yaml
  configs/experiments/mimic_iii/random_mixup.yaml
  configs/experiments/mimic_iii/comal_mixup.yaml
)

for index in "${!CONFIGS[@]}"; do
  config="${CONFIGS[$index]}"
  method="${METHODS[$index]}"
  echo "=== ${ACTION}: ${config} ==="
  if [[ "$ACTION" == "active" ]]; then
    python -m al_mimic.cli active --task mimic_iii --method "$method" --config "$config"
  else
    python -m al_mimic.cli "$ACTION" --task mimic_iii --config "$config"
  fi
done

if [[ "$ACTION" == "active" ]]; then
  for config in "${CONFIGS[@]}"; do
    python -m al_mimic.cli visualize --task mimic_iii --config "$config"
  done
  python scripts/evaluation/compare_experiments.py \
    --experiment experiments/mimic_iii_yang_wu_bertencoder_random Random \
    --experiment experiments/mimic_iii_yang_wu_bertencoder_comal CoMAL \
    --experiment experiments/mimic_iii_yang_wu_bertencoder_mm_comal MM-CoMAL \
    --experiment experiments/mimic_iii_yang_wu_bertencoder_modis MoDIS \
    --experiment experiments/mimic_iii_yang_wu_bertencoder_mosaic MoSAIC \
    --experiment experiments/mimic_iii_yang_wu_bertencoder_random_mixup Random+Mixup \
    --experiment experiments/mimic_iii_yang_wu_bertencoder_comal_mixup CoMAL+Mixup \
    --output experiments/yang_wu_random_and_mixup_recall_at_30.png
fi

echo RANDOM_AND_MIXUP_DONE
