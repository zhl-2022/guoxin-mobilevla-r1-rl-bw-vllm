#!/usr/bin/env python3
"""Audit004 standard seven-case raw HTTP pilot; no model/browser/GPU imports."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import zipfile

from audit_mobilevla_comparison004_007 import checked_archive, digest, require, EXPECTED_TOOLS_SHA

ROOT = Path(__file__).resolve().parents[1]
CASES = ["health", "models", "single_image", "multiple_images", "multiturn_first", "multiturn_followup", "streaming_image"]


def audit(payload, prefix, cleanup_name, tools):
    raw = lambda name: payload[prefix + "/" + name]
    read = lambda name: json.loads(raw(name))
    result, config, state, env = (read(name) for name in ("pilot-result.json", "config.json", "state.json", "env.log"))
    run_id = result["run_id"]
    require(re.fullmatch("[0-9a-f]{32}", run_id) is not None and config["run_id"] == state["run_id"] == run_id and raw("run-id.txt").decode().strip() == run_id, "Pilot owner identity differs")
    require(result["controller_exit_code"] == result["stop_exit_code"] == 0, "Pilot/stop did not exit zero")
    for name, row in result["files"].items():
        require(len(raw(name)) == row["bytes"] and digest(raw(name)) == row["sha256"], "Pilot raw seal differs: " + name)
    remote = str(Path(config["task"]).parent.parent).replace("\\", "/") + "/"
    require(remote == "/root/private_data/guoxin-mobilevla-tools-007-004/", "Pilot is not the independent004 tool directory")
    require(len(result["source_sha256"]) == 9, "Unexpected004 module/supervisor/smoke source set")
    sources = {}
    for name, sha in result["source_sha256"].items():
        require(name.startswith(remote), "Pilot source outside004")
        relative = name[len(remote):]
        require(relative in tools and digest(tools[relative]) == sha, "Pilot executed source differs from004: " + relative)
        sources[relative] = sha
    command = config["command"]
    require(command[command.index("--generation-config") + 1] == "vllm" and config["model_path"].endswith("/mobilevla-r1-derived007-eos004"), "004 generation config/derived path differs")
    require(config["architecture"] == "MobileVLAForConditionalGeneration006" and config["precision"] == "bf16" and config["tp"] == 1 and config["max_len"] == 4096, "Native model/profile differs")
    require(state["command"] == command and state["status"] == "stopped" and state["reason"] == "stop requested", "Final state not owned/stopped")
    require(env["configuration"] == {k: v for k, v in config.items() if k != "run_id"}, "Environment/config snapshot differs")
    require(env["gpu_available"] is True and env["devices"] == ["BW"] and env["hy-smi"]["code"] == 0, "Actual BW runtime evidence missing")
    whole_server = raw("server.log").decode("utf-8")
    marker = re.search(r"^=== run " + run_id + r" at .+ ===$", whole_server, re.M)
    require(marker is not None, "Current run log marker missing")
    later = re.search(r"^=== run [0-9a-f]{32} at ", whole_server[marker.end():], re.M)
    server = whole_server[marker.start():marker.end() + later.start() if later else len(whole_server)]
    require(json.dumps(command) in server, "Current run command is not bound to server log")
    for text in ("Platform plugin hcu is activated", "Resolved architecture: MobileVLAForConditionalGeneration006", "Loading weights took", "quantization=None", "device_config=cuda", "Shutdown complete", "Application shutdown complete."):
        require(text in server, "Current run server marker missing: " + text)
    require("Following weights were not initialized" not in server and "Traceback (most recent call last)" not in server, "Unresolved runtime/strict loader failure")
    groups = [int(x) for x in re.findall(r"Loading safetensors checkpoint shards:\s+0% Completed \| 0/(\d+)", server)]
    require(groups == [4, 1, 1], "003 checkpoint shard topology differs")
    posts = server.count('"POST /v1/chat/completions HTTP/1.1" 200 OK')
    require(posts == 5, "Expected five actual chat HTTP200 access records")
    rows = [json.loads(line) for line in raw("smoke.log").decode("utf-8").splitlines() if line.strip()]
    require([row["case"] for row in rows] == ["scope", *CASES, "summary"], "Seven-case raw smoke membership differs")
    ids = {row["run_id"] for row in rows}
    require(len(ids) == 1 and re.fullmatch("[0-9a-f]{32}", next(iter(ids))) is not None, "Mixed smoke client identity")
    require(rows[0]["official_accuracy"] is False and rows[-1]["functional_checks_passed"] is True and rows[-1]["results"] == [{"case": case, "passed": True} for case in CASES], "Scope/summary differs")
    require(datetime.fromisoformat(state["ready_utc"]) <= datetime.fromisoformat(rows[0]["utc"]) <= datetime.fromisoformat(rows[-1]["utc"]) <= datetime.fromisoformat(state["ended_utc"]), "Smoke requests outside owned service lifetime")
    for image in rows[0]["images"]:
        require(image["path"].startswith(remote), "Fixture outside004")
        relative = image["path"][len(remote):]
        require(digest(tools[relative]) == image["sha256"] and len(tools[relative]) == image["bytes"], "Fixture bytes differ")
    cases, truncated, nonpublisher_stops = {}, [], []
    for row in rows[1:-1]:
        case = row["case"]
        require(row["passed"] is True, "Functional smoke failed: " + case)
        if case in {"health", "models"}:
            require(row["response"]["status"] == 200, "GET status differs")
            if case == "models":
                require(any(x["id"] == config["served_name"] and x["root"] == config["model_path"] for x in json.loads(row["response"]["body"])["data"]), "Served model identity differs")
            cases[case] = {"passed": True, "http_status": 200}
        elif case == "streaming_image":
            text = "".join(choice.get("delta", {}).get("content", "") for event in row["events"] for choice in event["choices"])
            require(row["done"] is True and row["text"] == text and bool(text.strip()) and not row["stream_errors"], "SSE output incomplete")
            require(row["finish_reasons"] and all(x in {"stop", "length"} for x in row["finish_reasons"]), "SSE finish reason invalid")
            require(row["answer_truncated"] == ("length" in row["finish_reasons"]), "SSE truncation flag differs")
            stop_reasons = [choice.get("stop_reason") for event in row["events"] for choice in event["choices"] if choice.get("finish_reason") is not None]
            cases[case] = {"passed": True, "text": text, "duration_seconds": row["duration"], "finish_reasons": row["finish_reasons"], "stop_reasons": stop_reasons, "SSE_done": True, "events": len(row["events"]), "answer_truncated": row["answer_truncated"]}
            if 128001 in stop_reasons:
                nonpublisher_stops.append(case)
        else:
            response = row["response"];choice = response["choices"][0];text = choice["message"]["content"]
            require(response["model"] == config["served_name"] and isinstance(text, str) and bool(text.strip()), "Nonstream content missing")
            require(choice["finish_reason"] == row["finish_reason"] and row["finish_reason"] in {"stop", "length"} and row["answer_truncated"] == (row["finish_reason"] == "length"), "Generation completion/truncation differs")
            usage = response["usage"]
            require(usage["prompt_tokens"] > 0 and 0 < usage["completion_tokens"] <= 128 and usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], "Smoke usage accounting differs")
            cases[case] = {"passed": True, "text": text, "duration_seconds": row["duration"], "finish_reason": row["finish_reason"], "stop_reason": choice.get("stop_reason"), "answer_truncated": row["answer_truncated"], "usage": usage}
            if choice.get("stop_reason") == 128001:
                nonpublisher_stops.append(case)
        if row.get("answer_truncated"):
            truncated.append(case)
    require(not nonpublisher_stops, "004 pilot stopped at nonpublisher128001")
    cleanup = json.loads(payload[cleanup_name])
    require(cleanup.get("run_id") == run_id and set(cleanup["processes"]) == {"supervisor", "server"}, "Independent cleanup owner/schema differs")
    for role in ("supervisor", "server"):
        process = cleanup["processes"][role]
        require(process["pid"] == state[role + "_pid"] and process["expected_start_token"] == state[role + "_token"] and process["proc_exists"] is False, "Independent cleanup owner differs: " + role)
    require(cleanup["port30139closed"] is True and datetime.fromisoformat(cleanup["UTC"]) > datetime.fromisoformat(result["ended_utc"]), "Independent cleanup port/time differs")
    loading = re.search(r"Loading weights took ([0-9.]+) seconds", server)
    return {"schema": "mobilevla-standard004-pilot-independent-audit007-v1", "passed": True, "run_id": run_id,
            "smoke_client_run_id": next(iter(ids)), "source_files_bound_to_frozen004": sources,
            "HTTP_checks": cases, "functional_checks_passed": True, "truncated_cases": truncated,
            "observed_stop_reason128001_cases": nonpublisher_stops,
            "publisher_stop_semantics_consistent": None if not nonpublisher_stops else False,
            "publisher_EOS128009_triggered_verified": False,
            "EOS_scope": "No raw128001 stop is allowed. With terminal raw IDs absent, null primary stop_reason alone cannot directly identify EOS. Actual128009 trigger is proved separately by the same-source long native/HF raw-ID comparison, if observed.",
            "observed_shard_groups": groups, "recorded_weight_load_seconds": float(loading[1]),
            "runtime_environment": {k: env[k] for k in ("python", "packages", "devices", "hip", "gpu_available")},
            "state_stopped_and_owner_bound": True, "independent_cleanup": cleanup,
            "warnings": {"tokenizer_incorrect_regex": "incorrect regex pattern" in server,
                         "ROCm_GELU_fallback": "Falling back to GELU(approximate='none')" in server,
                         "destroy_process_group_not_explicit": "destroy_process_group() was not called" in server},
            "navigation_accuracy_verified": False, "official_accuracy_verified": False, "official_performance_verified": False,
            "official_acceptance": False, "submitted": False,
            "scope": "Downloaded004 standard pilot records audited locally; transport/nonempty output only, not semantic scoring or fresh GPU execution"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--prefix")
    parser.add_argument("--cleanup")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Audit output must be new")
    payload, identity = checked_archive(args.archive)
    candidates = [name.rsplit("/", 1)[0] for name in payload if name.endswith("/pilot-result.json")]
    prefix = args.prefix or (candidates[0] if len(candidates) == 1 else None)
    require(bool(prefix), "Select a unique pilot prefix")
    cleanups = [name for name in payload if "pilot004" in name and name.endswith("independent-cleanup.json")]
    cleanup = args.cleanup or (cleanups[0] if len(cleanups) == 1 else None)
    require(bool(cleanup), "Select a unique pilot004 independent cleanup record")
    tools_path = ROOT / "dist/session007-mobilevla-tools-004.zip"
    require(digest(tools_path.read_bytes()) == EXPECTED_TOOLS_SHA, "Expected exact003 tool archive")
    with zipfile.ZipFile(tools_path) as archive:
        tools = {name: archive.read(name) for name in archive.namelist()}
    result = audit(payload, prefix, cleanup, tools)
    result.update(raw_archive=identity, raw_prefix=prefix, cleanup_member=cleanup)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
