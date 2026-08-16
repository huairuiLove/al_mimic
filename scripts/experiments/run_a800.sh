#!/usr/bin/env bash
# Device-local entrypoint for the current A800 + 18-core-quota host.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_a800_144c.sh" "$@"
