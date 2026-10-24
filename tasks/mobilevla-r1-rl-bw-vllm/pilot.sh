#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$TASK_DIR/env.sh"
RUN_ID="$("$PYTHON_BIN" -c 'import secrets; print(secrets.token_hex(16))')"
OUTPUT="$TASK_DIR/pilot-evidence/$RUN_ID"
mkdir -p "$TASK_DIR/pilot-evidence"
mkdir "$OUTPUT"
printf '%s\n' "$RUN_ID" > "$OUTPUT/run-id.txt"
finish() {
  local STATUS=$?
  local STOP_STATUS=0
  trap - EXIT INT TERM
  bash "$TASK_DIR/stop.sh" --run-id "$RUN_ID" > "$OUTPUT/stop.log" 2>&1 || STOP_STATUS=$?
  if [[ "$STATUS" == 0 && "$STOP_STATUS" != 0 ]]; then STATUS="$STOP_STATUS"; fi
  "$PYTHON_BIN" -B - "$TASK_DIR" "$OUTPUT" "$RUN_ID" "$STATUS" "$STOP_STATUS" <<'PY'
import datetime,hashlib,json,sys
from pathlib import Path
task,out,run_id,status=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],int(sys.argv[4])
state=task/'runtime/state.json'
if state.is_file() and json.loads(state.read_text()).get('run_id')==run_id:
    for name,path in [('state.json',state),('config.json',task/f'runtime/config-{run_id}.json'),('server.log',task/'runtime/server.log'),('supervisor.log',task/'runtime/supervisor.log')]:
        if path.is_file() and path.stat().st_size<=50*1024*1024:(out/name).write_bytes(path.read_bytes())
project=task.parent.parent
scripts=list((project/'candidates/mobilevla-r1-session006/mobilevla_bw006').glob('*.py'))+[project/'scripts/recon2reason_runtime.py',project/'scripts/recon2reason_smoke.py']
record={'run_id':run_id,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'controller_exit_code':status,'stop_exit_code':int(sys.argv[5]),'source_sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts},'files':{p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in out.iterdir() if p.is_file()},'official_accuracy_verified':False,'official_performance_verified':False,'submission_ready':False,'scope':'Raw execution only; inspect response/server/owned cleanup before interpreting result'}
with (out/'pilot-result.json').open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
PY
  printf 'MobileVLA pilot archive: %s (controller exit %s)\n' "$OUTPUT" "$STATUS"
  exit "$STATUS"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
bash "$TASK_DIR/run.sh" --run-id "$RUN_ID" > "$OUTPUT/start.log" 2>&1
bash "$TASK_DIR/env_check.sh" > "$OUTPUT/env.log" 2>&1
timeout --signal=TERM --kill-after=15s 600 bash "$TASK_DIR/smoke.sh" > "$OUTPUT/smoke.log" 2>&1
