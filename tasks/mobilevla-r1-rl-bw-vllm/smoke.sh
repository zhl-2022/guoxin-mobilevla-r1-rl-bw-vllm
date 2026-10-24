#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$TASK_DIR/env.sh"
exec "$PYTHON_BIN" -B "$TASK_DIR/../../scripts/recon2reason_smoke.py" --base-url "http://127.0.0.1:$MOBILE_PORT" --model "$SERVED_NAME" --log "$TASK_DIR/smoke.log" --timeout 180 --max-tokens 128 --image "$TASK_DIR/fixtures/red.png" --image "$TASK_DIR/fixtures/blue.png" "$@"
