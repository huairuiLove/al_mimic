#!/usr/bin/env bash
# Device-local entrypoint for the current A800 + 18-core-quota host.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_a800_144c.sh" "$@"
