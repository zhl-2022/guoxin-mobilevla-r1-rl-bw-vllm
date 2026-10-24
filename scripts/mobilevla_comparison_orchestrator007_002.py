#!/usr/bin/env python3
"""Owned native -> verified cleanup -> HF comparison, within 1800 seconds.

Separate from frozen 006-002. Only its own new run ID is stopped. All outputs
are new files under private_data; subprocesses and cleanup have bounded waits.
"""
from __future__ import annotations

import argparse
import datetime
import errno
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time

LOCK_SHA256 = "f27db0276bc887eeb0c14f5e9deff9af40657017c2caa00cb07d596ac2ef33b9"
PRIVATE_ROOT = Path("/root/private_data")


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fresh_json(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def regular_ancestors(path):
    path = Path(path).absolute()
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("Symlink paths are not accepted: " + str(path))
    if path.resolve(strict=False) != path:
        raise ValueError("Paths must be canonical without parent traversal: " + str(path))
    return path


def check_sources(frozen, comparison, lock_path=None):
    lock_path = lock_path or Path(__file__).with_name("mobilevla_orchestrator007_002-lock.json")
    if sha(lock_path) != LOCK_SHA256:
        raise ValueError("Orchestrator source lock identity differs")
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    counts = {}
    for kind, root in (("frozen", frozen), ("comparison", comparison)):
        for name, expected in lock[kind]["files"].items():
            path = regular_ancestors(Path(root) / name)
            if not path.is_file() or path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
                raise ValueError("Sealed source differs: " + kind + "/" + name)
        counts[kind] = len(lock[kind]["files"])
    return {"status": "PASS", "verified_payload_file_counts": counts, "lock_sha256": LOCK_SHA256}


def proc_identity(pid):
    if type(pid) is not int or pid <= 0:
        return {"status": "invalid_pid", "pid": pid, "running": None}
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"status": "observed", "pid": pid, "state": fields[0], "start_token": fields[19],
                "process_group": int(fields[2]), "running": fields[0] != "Z"}
    except FileNotFoundError:
        return {"status": "absent", "pid": pid, "running": False}
    except (OSError, IndexError, ValueError) as exc:
        return {"status": "unknown", "pid": pid, "running": None, "error": str(exc)}


def verify_owned_cleanup(state, run_id):
    if state.get("run_id") != run_id:
        return {"status": "UNKNOWN", "reason": "state no longer belongs to requested run; no other run is signalled"}
    observed = {}
    owned_gone = True
    for role in ("supervisor", "server"):
        item = proc_identity(state.get(role + "_pid"))
        item["expected_start_token"] = state.get(role + "_token")
        item["same_owned_process_running"] = (item.get("running") is True and
                                                item.get("start_token") == state.get(role + "_token"))
        if item.get("running") is None or not state.get(role + "_token") or item["same_owned_process_running"]:
            owned_gone = False
        observed[role] = item
    group = state.get("server_pid")
    group_live, group_errors = [], []
    if type(group) is int and group > 0:
        for entry in Path("/proc").iterdir():
            if entry.name.isdecimal():
                item = proc_identity(int(entry.name))
                if item.get("status") == "unknown":
                    group_errors.append(item)
                elif item.get("process_group") == group and item.get("running"):
                    group_live.append(item)
    else:
        group_errors.append({"reason": "missing server process group identity"})
    with socket.socket() as stream:
        stream.settimeout(2)
        port_result = stream.connect_ex(("127.0.0.1", 30139))
    port_closed = port_result == errno.ECONNREFUSED
    terminal = state.get("status") in {"stopped", "failed", "expired"} and bool(state.get("ended_utc"))
    passed = terminal and owned_gone and not group_live and not group_errors and port_closed
    return {"status": "PASS" if passed else "FAIL", "captured_utc": stamp(), "run_id": run_id,
            "terminal_state_recorded": terminal, "processes": observed,
            "owned_process_group_live_members": group_live, "process_probe_errors": group_errors,
            "loopback_port": 30139, "connect_ex": port_result, "port_refused_connection": port_closed,
            "GPU_memory_idle_claimed": False}


class Controller:
    def __init__(self, output, deadline, environment):
        self.output, self.deadline, self.environment = Path(output), deadline, environment
        self.steps = []

    def execute(self, name, command, cap, allowed=(0,), cleanup=False):
        remaining = self.deadline - time.monotonic() - (0 if cleanup else 65)
        timeout = min(float(cap), remaining)
        if timeout <= 0:
            raise TimeoutError("Total deadline reached before " + name)
        step = {"name": name, "command": [str(x) for x in command], "started_utc": stamp(),
                "timeout_seconds": timeout, "status": "running"}
        self.steps.append(step)
        began = time.monotonic()
        process = None
        try:
            with (self.output / (name + ".log")).open("x", encoding="utf-8") as stream:
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                           stderr=subprocess.STDOUT, env=self.environment, start_new_session=True)
                try:
                    step["returncode"] = process.wait(timeout=max(0.1, timeout - 5))
                except subprocess.TimeoutExpired:
                    step["timed_out"] = True
                    self.kill_group(process)
                    step["returncode"] = process.returncode
                    raise TimeoutError("Bounded subprocess timed out: " + name)
            step["status"] = "PASS" if step["returncode"] in allowed else "FAIL"
            if step["status"] != "PASS":
                raise RuntimeError(f"{name} exited {step['returncode']}")
            return step["returncode"]
        finally:
            if process is not None and process.poll() is None:
                self.kill_group(process)
            step["elapsed_seconds"] = time.monotonic() - began
            if step["status"] == "running":
                step["status"] = "FAIL"
            fresh_json(self.output / (name + "-step.json"), step)

    @staticmethod
    def kill_group(process):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=2)


def stop_owned(controller, frozen, run_id, label):
    task = Path(frozen) / "tasks/mobilevla-r1-rl-bw-vllm"
    state_path = task / "runtime/state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    if state.get("run_id") != run_id:
        result = {"status": "UNKNOWN", "run_id": run_id,
                  "reason": "requested run not present; no unrelated process was signalled"}
    else:
        stop_error = None
        try:
            controller.execute(label, ["bash", str(task / "stop.sh"), "--run-id", run_id], 45, cleanup=True)
        except Exception as exc:
            stop_error = f"{type(exc).__name__}: {exc}"
        state = json.loads(state_path.read_text())
        result = verify_owned_cleanup(state, run_id)
        result["stop_error"] = stop_error
        if stop_error:
            result["status"] = "FAIL"
        fresh_json(controller.output / (label + "-state.json"), state)
    fresh_json(controller.output / (label + "-verification.json"), result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, default=Path("/root/private_data/guoxin-mobilevla-tools-007-003"))
    parser.add_argument("--comparison", type=Path, default=Path("/root/private_data/guoxin-mobilevla-tools-007-003"))
    parser.add_argument("--source", type=Path, default=Path("/root/private_data/guoxin-candidates/models/mobilevla-r1-rl006/weight/rl"))
    parser.add_argument("--model", type=Path, default=Path("/root/private_data/guoxin-candidates/models/mobilevla-r1-derived007-bos003"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--total-timeout", type=int, default=1800)
    args = parser.parse_args(argv)
    if sys.platform != "linux" or not 180 <= args.total_timeout <= 1800:
        raise ValueError("Actual orchestration requires Linux and a total budget of 180-1800 seconds")
    began = time.monotonic()
    for path in (args.output, args.frozen, args.comparison, args.source, args.model):
        regular_ancestors(path)
        if not path.absolute().is_relative_to(PRIVATE_ROOT):
            raise ValueError("Source and output paths must stay within /root/private_data")
    if args.output.exists() or not args.output.parent.is_dir():
        raise FileExistsError("Output must be a new directory with an existing parent")
    source_check = check_sources(args.frozen, args.comparison)
    run_id = secrets.token_hex(16)
    args.output.mkdir()
    fresh_json(args.output / "orchestrator-plan.json", {"schema": "mobilevla-owned-comparison-orchestrator007-v2",
               "run_id": run_id, "started_utc": stamp(), "total_timeout_seconds": args.total_timeout,
               "source_identity_before": source_check, "source": str(args.source), "model": str(args.model),
               "frozen": str(args.frozen), "comparison": str(args.comparison)})
    environment = dict(os.environ)
    environment.update(MOBILE_PYTHON_BIN=args.python, MOBILE_MODEL_PATH=str(args.model), MOBILE_PORT="30139",
                       MOBILE_LOAD_TIMEOUT=str(min(650, args.total_timeout - 120)),
                       MOBILE_MAX_RUNTIME_SECONDS=str(min(1100, args.total_timeout - 65)),
                       PYTHONPATH=str(args.frozen / "candidates/mobilevla-r1-session006"),
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    controller = Controller(args.output, began + args.total_timeout, environment)
    record = {"run_id": run_id, "status": "failed", "official_accuracy_verified": False,
              "official_performance_verified": False, "submitted": False}
    task = args.frozen / "tasks/mobilevla-r1-rl-bw-vllm"
    attempted_launch = False
    def interrupted(signum, _frame):
        raise KeyboardInterrupt("Controller interrupted by signal " + str(signum))
    old_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        if not args.model.exists():
            controller.execute("prepare-derived", [args.python, "-B", "-m", "mobilevla_bw006.prepare",
                 "--source", str(args.source), "--receipt", str(args.source.parents[1] / "extraction-receipt.json"),
                 "--output", str(args.model)], 240)
        receipt = json.loads((args.model / "prepare-receipt.json").read_text())
        if (receipt.get("source") != str(args.source) or receipt.get("derived") != str(args.model)
                or receipt.get("schema") != "mobilevla-derived-hf-view-007-bos003-v1"
                or receipt.get("adapter_revision") != "007-bos003"):
            raise ValueError("Derived view does not belong to this 003 source and adapter")
        fresh_json(args.output / "derived-prepare-receipt.json", receipt)
        controller.execute("native-tokenizer-probe", [args.python, "-B", str(args.comparison / "scripts/probe_mobilevla_tokens007_003.py"),
             "--source", str(args.source), "--derived", str(args.model),
             "--output", str(args.output / "native-tokenizer-probe.json")], 90)
        attempted_launch = True
        try:
            controller.execute("native-start", ["bash", str(task / "run.sh"), "--run-id", run_id], 700)
            state = json.loads((task / "runtime/state.json").read_text())
            if state.get("run_id") != run_id or state.get("status") != "ready":
                raise RuntimeError("Native readiness does not belong to this new run")
            fresh_json(args.output / "native-ready-state.json", state)
            config = json.loads((task / "runtime" / ("config-" + run_id + ".json")).read_text())
            fresh_json(args.output / "native-config.json", config)
            for frames in (1, 8):
                image_args = []
                for index in range(frames):
                    image_args.extend(["--image", str(task / "fixtures" / ("red.png" if index % 2 == 0 else "blue.png"))])
                controller.execute("native-" + str(frames), [args.python, "-B", str(args.comparison / "scripts/mobilevla_reference_http006_003.py"),
                     "--source", str(args.source), "--max-new-tokens", "32", *image_args,
                     "--output", str(args.output / f"native-{frames}.json")], 245)
        finally:
            record["native_cleanup"] = stop_owned(controller, args.frozen, run_id, "native-stop")
        if record["native_cleanup"]["status"] != "PASS":
            raise RuntimeError("Independent owned cleanup did not pass; HF reference is blocked")
        if shutil.which("hy-smi"):
            controller.execute("after-native-hy-smi", ["hy-smi"], 20)
        for frames in (1, 8):
            image_args = []
            for index in range(frames):
                image_args.extend(["--image", str(task / "fixtures" / ("red.png" if index % 2 == 0 else "blue.png"))])
            controller.execute("hf-" + str(frames), [args.python, "-B", str(args.comparison / "scripts/mobilevla_reference_hf006_003.py"),
                 "--source", str(args.source), "--max-new-tokens", "32", *image_args,
                 "--output", str(args.output / f"hf-{frames}.json")], 480)
        comparisons = []
        for frames in (1, 8):
            controller.execute("compare-" + str(frames), [args.python, "-B", str(args.comparison / "scripts/mobilevla_compare_records006_003.py"),
                 "--native", str(args.output / f"native-{frames}.json"), "--reference", str(args.output / f"hf-{frames}.json"),
                 "--output", str(args.output / f"comparison-{frames}.json")], 15, allowed=(0, 2))
            comparisons.append(json.loads((args.output / f"comparison-{frames}.json").read_text()))
        record["comparisons"] = comparisons
        record["status"] = ("completed_equal_raw_tokens" if all(x["comparable"] and x["raw_generated_token_ids_equal"] for x in comparisons)
                            else "completed_with_divergence_or_incomparable_records")
    except BaseException as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        if attempted_launch:
            try:
                record["final_cleanup"] = stop_owned(controller, args.frozen, run_id, "final-stop")
            except BaseException as exc:
                record["final_cleanup"] = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
        for name in ("state.json", "server.log", "supervisor.log"):
            path = task / "runtime" / name
            if path.is_file() and path.stat().st_size <= 50 * 1024 * 1024:
                target = args.output / ("native-runtime-" + name)
                with target.open("xb") as stream:
                    stream.write(path.read_bytes())
        try:
            record["source_identity_after"] = check_sources(args.frozen, args.comparison)
        except Exception as exc:
            record["source_identity_after"] = {"status": "FAIL", "error": str(exc)}
        if attempted_launch and record.get("final_cleanup", {}).get("status") != "PASS":
            record["status"] = "failed_owned_cleanup"
        elif record["source_identity_after"]["status"] != "PASS":
            record["status"] = "failed_source_identity_after"
        record.update(ended_utc=stamp(), elapsed_seconds=time.monotonic() - began, steps=controller.steps)
        record["files"] = {path.name: {"bytes": path.stat().st_size, "sha256": sha(path)}
                           for path in args.output.iterdir() if path.is_file()}
        fresh_json(args.output / "orchestrator-result.json", record)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    print(json.dumps({"output": str(args.output), "run_id": run_id, "status": record["status"],
                      "final_cleanup": record.get("final_cleanup"), "error": record.get("error")}, ensure_ascii=False))
    return 0 if record["status"] == "completed_equal_raw_tokens" and record.get("final_cleanup", {}).get("status") == "PASS" and record["source_identity_after"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
