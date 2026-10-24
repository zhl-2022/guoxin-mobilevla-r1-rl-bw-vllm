#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
exec bash "$HERE/../mobilevla-r1-rl-bw-vllm/env_check.sh" "$@"
