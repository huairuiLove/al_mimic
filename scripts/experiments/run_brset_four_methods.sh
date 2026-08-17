#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BASE_CONFIG="configs/experiments/brset/comal.yaml"
python -m al_mimic.cli prepare --task brset --config "$BASE_CONFIG"
python -m al_mimic.cli validate-data --task brset --config "$BASE_CONFIG"

for method in comal mm_comal modis mosaic; do
  config="configs/experiments/brset/${method}.yaml"
  python -m al_mimic.cli active --task brset --method "$method" --config "$config"
done
