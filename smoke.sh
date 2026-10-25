#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/tasks/mobilevla-r1-rl-bw-vllm-standard007/smoke.sh" "$@"
