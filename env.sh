#!/usr/bin/env bash
# Resolve the canonical task before sourcing its sealed environment.
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/tasks/mobilevla-r1-rl-bw-vllm" && pwd)"
source "$TASK_DIR/env.sh"
