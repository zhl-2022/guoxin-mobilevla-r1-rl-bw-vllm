#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$TASK_DIR/env.sh"
exec "$PYTHON_BIN" -B -m mobilevla_bw006.runtime verify --task-dir "$TASK_DIR" "$@"
