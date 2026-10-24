#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$#" != 2 || "$1" != --run-id || ! "$2" =~ ^[0-9a-f]{32}$ ]]; then
  printf 'Supply --run-id followed by the exact 32 hex owned run ID\n' >&2
  exit 2
fi
exec bash "$HERE/../mobilevla-r1-rl-bw-vllm/stop.sh" "$@"
