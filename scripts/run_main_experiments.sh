#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-all}"
CONFIG="${CONFIG:-configs/mimic_a800_144c.yaml}"

case "$ACTION" in
  prepare)
    python main.py prepare --config "$CONFIG"
    python main.py validate-data --config "$CONFIG"
    python main.py explore --config "$CONFIG"
    ;;
  features)
    python main.py features --config "$CONFIG"
    ;;
  active)
    scripts/run_four_methods.sh active
    ;;
  all)
    "$0" prepare
    "$0" features
    "$0" active
    ;;
  *)
    echo "usage: $0 {prepare|features|active|all}" >&2
    exit 2
    ;;
esac
