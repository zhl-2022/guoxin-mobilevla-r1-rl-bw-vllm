#!/usr/bin/env bash
# Standard entry points delegate to the source-sealed native task.
export MOBILE_MODEL_PATH="${MOBILE_MODEL_PATH:-/root/private_data/guoxin-candidates/models/mobilevla-r1-derived007-eos004}"
export MOBILE_PORT="${MOBILE_PORT:-30139}"
export MOBILE_MEM_FRAC="${MOBILE_MEM_FRAC:-0.80}"
export MOBILE_LOAD_TIMEOUT="${MOBILE_LOAD_TIMEOUT:-900}"
export MOBILE_MAX_RUNTIME_SECONDS="${MOBILE_MAX_RUNTIME_SECONDS:-1800}"
