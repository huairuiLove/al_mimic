#!/usr/bin/env bash
set -euo pipefail

python -m al_mimic.tasks.mimic_iii.preprocessing.extract_notes "$@"
