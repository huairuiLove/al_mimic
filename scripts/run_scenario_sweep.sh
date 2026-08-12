#!/usr/bin/env bash
# Two-phase sweep for one scenario: prove it can resolve strategies, then run them.
#
# Phase 1 trains only the random arm and the full-pool ceiling, then asks whether
# the ceiling is far enough above random for any acquisition difference to clear
# test noise. On the official cohort that gap is 0.0117 against a noise
# half-width of 0.0063, i.e. 1.9x, which is why the six arms already run came
# back indistinguishable. Phase 2 runs only if phase 1 passes.
#
# Usage: run_scenario_sweep.sh <scenario-suffix> [--force]
#   scenario-suffix names the config family, e.g. missing_notes or mid_labels,
#   matching configs/mimic_<strategy>_<suffix>.yaml

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SUFFIX="${1:?usage: run_scenario_sweep.sh <scenario-suffix> [--force]}"
FORCE="${2:-}"

RANDOM_CONFIG="configs/mimic_random_${SUFFIX}.yaml"
RANDOM_DIR="experiments/mimic_iii_yang_wu_bertencoder_random_${SUFFIX}"
CEILING_DIR="${RANDOM_DIR}_full_pool"

echo "=== phase 1: random arm (${SUFFIX}) ==="
python -u main.py active --config "$RANDOM_CONFIG"

echo "=== phase 1: full-pool ceiling (${SUFFIX}) ==="
python -u scripts/run_full_pool_ceiling.py --config "$RANDOM_CONFIG"

echo "=== phase 1: resolution test (${SUFFIX}) ==="
set +e
python -u scripts/resolution_test.py \
  --random "$RANDOM_DIR" --ceiling "$CEILING_DIR" --draws 2000
VERDICT=$?
set -e

if [[ $VERDICT -ne 0 && "$FORCE" != "--force" ]]; then
  echo "Stopping before the method sweep: this scenario cannot resolve strategies."
  echo "Adjust the scenario, or re-run with --force to spend the GPU time anyway."
  exit $VERDICT
fi

echo "=== phase 2: method sweep (${SUFFIX}) ==="
for strategy in comal mm_comal modis mosaic; do
  config="configs/mimic_${strategy}_${SUFFIX}.yaml"
  if [[ ! -f "$config" ]]; then
    echo "skip ${strategy}: ${config} not present"
    continue
  fi
  echo "--- ${strategy} ---"
  python -u main.py active --config "$config"
done

echo "=== scoring (${SUFFIX}) ==="
ARMS=()
for strategy in comal mm_comal modis mosaic; do
  dir="experiments/mimic_iii_yang_wu_bertencoder_${strategy}_${SUFFIX}"
  [[ -d "$dir" ]] && ARMS+=(--arm "$dir" "$strategy")
done
python -u scripts/resolution_test.py \
  --random "$RANDOM_DIR" --ceiling "$CEILING_DIR" --draws 2000 "${ARMS[@]}" || true

echo "SCENARIO_SWEEP_DONE ${SUFFIX}"
