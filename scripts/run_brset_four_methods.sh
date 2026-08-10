#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m brset_al.cli prepare --config configs/brset_comal.yaml
python -m brset_al.cli validate-data --config configs/brset_comal.yaml

CONFIGS=(
  configs/brset_comal.yaml
  configs/brset_mm_comal.yaml
  configs/brset_modis.yaml
  configs/brset_mosaic.yaml
)

for config in "${CONFIGS[@]}"; do
  python -m brset_al.cli active --config "$config"
done
