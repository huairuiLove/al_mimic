#!/usr/bin/env bash
# Serial queue for the acquisition control and the two mixup arms.
#
#   random        -- uniform sampling control; without it no claim about the
#                    informed strategies is falsifiable
#   random_mixup  -- isolates the mixup contribution from acquisition
#   comal_mixup   -- mixup on the arm that selects the sparsest visits
#
# Run this only when the GPU is free. It queues nothing and restarts nothing:
# a failing arm aborts the queue so the cause can be fixed before resuming.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ACTION="${1:-active}"
CONFIGS=(
  configs/mimic_random.yaml
  configs/mimic_random_mixup.yaml
  configs/mimic_comal_mixup.yaml
)

for config in "${CONFIGS[@]}"; do
  echo "=== ${ACTION}: ${config} ==="
  python -u main.py "$ACTION" --config "$config"
done

if [[ "$ACTION" == "active" ]]; then
  for config in "${CONFIGS[@]}"; do
    python -u main.py visualize --config "$config"
  done
  python -u scripts/compare_experiments.py \
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
