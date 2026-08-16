#!/usr/bin/env bash
# Gate a scenario method sweep on its random-to-full-data resolution.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SUFFIX="${1:?usage: run_scenario_sweep.sh <scenario-suffix> [--force]}"
FORCE="${2:-}"
RANDOM_CONFIG="configs/experiments/mimic_iii/random_${SUFFIX}.yaml"
RANDOM_DIR="experiments/mimic_iii_yang_wu_bertencoder_random_${SUFFIX}"
CEILING_DIR="${RANDOM_DIR}/full_data"

if [[ ! -f "$RANDOM_CONFIG" ]]; then
  echo "unknown scenario config: $RANDOM_CONFIG" >&2
  exit 2
fi

echo "=== phase 1: random arm (${SUFFIX}) ==="
python -m al_mimic.cli active \
  --task mimic_iii --method random --config "$RANDOM_CONFIG"

echo "=== phase 1: full-data ceiling (${SUFFIX}) ==="
python -m al_mimic.cli full-data --task mimic_iii --config "$RANDOM_CONFIG"

echo "=== phase 1: resolution test (${SUFFIX}) ==="
set +e
python scripts/evaluation/resolution_test.py \
  --random "$RANDOM_DIR" --ceiling "$CEILING_DIR" --draws 2000
VERDICT=$?
set -e

if [[ $VERDICT -ne 0 && "$FORCE" != "--force" ]]; then
  echo "Stopping before the method sweep: this scenario cannot resolve strategies."
  exit "$VERDICT"
fi

echo "=== phase 2: method sweep (${SUFFIX}) ==="
for method in comal mm_comal modis mosaic; do
  config="configs/experiments/mimic_iii/${method}_${SUFFIX}.yaml"
  if [[ ! -f "$config" ]]; then
    echo "skip ${method}: ${config} not present"
    continue
  fi
  python -m al_mimic.cli active --task mimic_iii --method "$method" --config "$config"
done

echo "=== scoring (${SUFFIX}) ==="
ARMS=()
for method in comal mm_comal modis mosaic; do
  directory="experiments/mimic_iii_yang_wu_bertencoder_${method}_${SUFFIX}"
  [[ -d "$directory" ]] && ARMS+=(--arm "$directory" "$method")
done
python scripts/evaluation/resolution_test.py \
  --random "$RANDOM_DIR" --ceiling "$CEILING_DIR" --draws 2000 "${ARMS[@]}" || true

echo "SCENARIO_SWEEP_DONE ${SUFFIX}"
