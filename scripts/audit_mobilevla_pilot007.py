"""Audit the preserved first MobileVLA BW pilot without loading model frameworks.

Copyright scope notice: no repository-wide open-source license is selected.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from build_auk_final007 import checked_zip
from verify_auk_final007_bundle import files_in

ROOT = Path(__file__).resolve().parents[1]
RAW_SHA = "f417ddaf51c2a312db922fd755fbefe53cbad3975002d4ba97149a4fc8898414"
TOOLS_SHA = "e116b10853b11fc81e5dcdc0dbd3c2d85fe36c4b8b38500a120a88c4552b92d2"
RUN_ID = "9c715239832bb23ab2270ab248ef0d8b"
REMOTE_TOOLS = "/root/private_data/guoxin-mobilevla-tools-006-002/"
CASES = ["health", "models", "single_image", "multiple_images",
         "multiturn_first", "multiturn_followup", "streaming_image"]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def manifest_check(payload):
    manifest = json.loads(payload["MANIFEST.json"])["files"]
    require(set(payload) == set(manifest) | {"MANIFEST.json"}, "Raw membership differs")
    for name, row in manifest.items():
        require(len(payload[name]) == row["bytes"] and sha(payload[name]) == row["sha256"],
                "Raw member identity differs: " + name)
    return manifest


def audit(raw, tools, original_receipt):
    read = lambda name: json.loads(raw[name])
    result = read("mobile-pilot001/pilot-result.json")
    config = read("mobile-pilot001/config.json")
    state = read("mobile-pilot001/state.json")
    env = read("mobile-pilot001/env.log")
    require(result["run_id"] == config["run_id"] == state["run_id"] == RUN_ID and
            raw["mobile-pilot001/run-id.txt"].decode().strip() == RUN_ID, "Owner identity differs")
    require(result["controller_exit_code"] == result["stop_exit_code"] == 0, "Controller/stop failed")
    for name, row in result["files"].items():
        data = raw["mobile-pilot001/" + name]
        require(len(data) == row["bytes"] and sha(data) == row["sha256"], "Result raw binding differs: " + name)
    require(len(result["source_sha256"]) == 9, "Unexpected runtime source set")
    sources = {}
    for name, expected in result["source_sha256"].items():
        require(name.startswith(REMOTE_TOOLS), "Unexpected source prefix")
        relative = name[len(REMOTE_TOOLS):]
        require(sha(tools[relative]) == expected, "Actual source differs from frozen002: " + relative)
        sources[relative] = expected
    require(config["model_revision"] == "0c757f3e9d1b95d8416b42e7089e1c04f3e6118a" and
            config["architecture"] == "MobileVLAForConditionalGeneration006" and
            config["precision"] == "bf16" and config["tp"] == 1 and config["max_num_seqs"] == 1,
            "Actual model/profile differs")
    require(config["contest_alias_equivalence_verified"] is False and config["GPU_verified"] is False,
            "Static config flags must remain original")
    require(state["command"] == config["command"] and state["url"] == config["url"] and
            state["status"] == "stopped" and state["reason"] == "stop requested", "Final state differs")
    require(env["configuration"] == {k: v for k, v in config.items() if k != "run_id"},
            "Environment config differs")
    require(env["gpu_available"] is True and env["devices"] == ["BW"] and
            env["hy-smi"]["code"] == 0 and env["hip"] == "6.3.26093", "Missing actual native BW evidence")

    server = raw["mobile-pilot001/server.log"].decode()
    for marker in ("Platform plugin hcu is activated", "Resolved architecture: MobileVLAForConditionalGeneration006",
                   "default_loader.py:384", "Loading weights took 3.52 seconds", "4/4", "1/1",
                   "device_config=cuda", "quantization=None", "hip/cutlassfa_adapter.h",
                   "Shutdown complete", "Application shutdown complete."):
        require(marker in server, "Raw service marker missing: " + marker)
    require(json.dumps(config["command"]) in server and "=== run " + RUN_ID + " at " in server,
            "Raw server is not command/owner-bound")
    require(server.count('"POST /v1/chat/completions HTTP/1.1" 200 OK') == 5,
            "Actual five generation POST records differ")
    shard_groups = [int(value) for value in re.findall(
        r"Loading safetensors checkpoint shards:\s+0% Completed \| 0/(\d+)", server)]
    require(shard_groups == [4, 1, 1], "Actual checkpoint shard group topology differs")
    require("Following weights were not initialized" not in server and "Traceback (most recent call last)" not in server,
            "Service includes an unresolved loader/runtime failure")
    backend = tools["candidates/mobilevla-r1-session006/mobilevla_bw006/backend.py"].decode()
    loader = tools["candidates/mobilevla-r1-session006/framework-source/vllm-model_executor-model_loader-default_loader.py"].decode()
    require("AutoWeightsLoader(self).load_weights(mapped)" in backend and
            "DefaultModelLoader.Source" in backend and "weights_not_loaded = weights_to_load - loaded_weights" in loader,
            "Frozen adapter/strict-loader contract differs")

    rows = [json.loads(line) for line in raw["mobile-pilot001/smoke.log"].decode().splitlines()]
    require([row["case"] for row in rows] == ["scope", *CASES, "summary"], "Smoke case membership differs")
    smoke_ids = {row["run_id"] for row in rows}
    require(len(smoke_ids) == 1, "Mixed smoke client owner identities")
    smoke_id = smoke_ids.pop()
    require(bool(re.fullmatch("[0-9a-f]{32}", smoke_id)), "Invalid smoke client ID")
    scope, summary = rows[0], rows[-1]
    require(scope["model"] == config["served_name"] and scope["official_accuracy"] is False and
            summary["functional_checks_passed"] is True and summary["official_accuracy"] is False,
            "Smoke scope/summary differs")
    require(summary["results"] == [{"case": case, "passed": True} for case in CASES], "Smoke summary differs")
    require(datetime.fromisoformat(state["ready_utc"]) <= datetime.fromisoformat(scope["utc"]) <=
            datetime.fromisoformat(summary["utc"]) <= datetime.fromisoformat(state["ended_utc"]),
            "Smoke requests are outside the owner service lifetime")
    for fixture in scope["images"]:
        name = fixture["path"][len(REMOTE_TOOLS):]
        require(fixture["path"].startswith(REMOTE_TOOLS) and sha(tools[name]) == fixture["sha256"] and
                len(tools[name]) == fixture["bytes"], "Smoke fixture differs from frozen002")
    cases = {}
    for row in rows[1:-1]:
        case = row["case"]
        require(row["passed"] is True, "Smoke check did not pass: " + case)
        if case in {"health", "models"}:
            require(row["response"]["status"] == 200, "Raw HTTP status differs: " + case)
            if case == "models":
                require(any(item["id"] == config["served_name"] for item in json.loads(row["response"]["body"])["data"]),
                        "Model list differs")
            cases[case] = {"passed": True, "http_status": 200}
        elif case != "streaming_image":
            reply = row["response"]
            content = reply["choices"][0]["message"]["content"]
            require(reply["model"] == config["served_name"] and bool(content.strip()) and
                    row["finish_reason"] == "stop" and row["answer_truncated"] is False, "Generation incomplete: " + case)
            cases[case] = {"passed": True, "duration_seconds": row["duration"], "text": content,
                           "finish_reason": row["finish_reason"], "usage": reply["usage"]}
        else:
            text = "".join(choice.get("delta", {}).get("content", "") for event in row["events"] for choice in event["choices"])
            require(row["done"] is True and row["text"] == text and bool(text.strip()) and
                    row["answer_truncated"] is False and row["stream_errors"] == [] and row["finish_reasons"] == ["stop"],
                    "Streaming events incomplete")
            cases[case] = {"passed": True, "duration_seconds": row["duration"], "text": text,
                           "SSE_done": True, "events": len(row["events"]), "stream_errors": []}

    cleanup = read("session007/mobile-pilot001-independent-cleanup.json")
    for role in ("supervisor", "server"):
        require(cleanup[role]["pid"] == state[role + "_pid"] and
                cleanup[role]["recorded_starttoken"] == state[role + "_token"] and
                cleanup[role]["proc_exists"] is False, "Independent owner cleanup differs: " + role)
    require(cleanup["port30139closed"] is True and
            datetime.fromisoformat(cleanup["UTC"]) > datetime.fromisoformat(result["ended_utc"]),
            "Independent port/time check differs")
    require("Stopped the matching task supervisor and its server process group." in raw["mobile-pilot001/stop.log"].decode(),
            "Raw owned stop response missing")

    proof = read("session007/mobile-covering-volume-proof001.json")
    volume_tools, volume_identity = checked_zip(ROOT / "dist/session006-mobilevla-volume-proof-001.zip")
    volume_lock_raw = volume_tools["evidence/session006/source-mapping/mobilevla-volume-sources006.json"]
    plan_raw = volume_tools["candidates/mobilevla-r1-session006/rl-entry-plan.json"]
    require(proof["source_sha256"] == {
        "downloader": sha(volume_tools["scripts/download_mobilevla_selected_006.py"]),
        "plan": sha(plan_raw), "volume_lock": sha(volume_lock_raw)}, "Volume proof source binding differs")
    require(sha(original_receipt) == proof["extraction_receipt_sha256"], "Original extraction receipt binding differs")
    receipt = json.loads(original_receipt)
    require(receipt["status"] == "completed" and receipt["full_lfs_parts_sha_verified"] is False and
            proof["all_26_full_volume_bytes_verified"] is False and proof["original_extraction_receipt_unchanged"] is True,
            "Full26 status or receipt immutability differs")
    lock = json.loads(volume_lock_raw)
    lookup, cursor = {}, 0
    for part in lock["parts"]:
        lookup[part["path"]] = {**part, "logical_start": cursor}
        cursor += part["size"]
    require(len(lookup) == 26 and len(proof["parts"]) == proof["covered_volumes_expected"] == 14 and
            len({part["path"] for part in proof["parts"]}) == 14 and
            proof["status"] == "completed_fourteen_covering_volumes_authenticated" and
            proof["all_selected_cache_SHA256_match_extraction_receipt"] is True,
            "Covering volume membership differs")
    caches = {row["path"]: row for row in proof["selected_cache_hashes"]}
    originals = {row["path"]: row for row in receipt["files"]}
    plan = json.loads(plan_raw)
    require(set(caches) == set(originals) == {row["path"] for row in plan["entries"]} and len(caches) == 16,
            "Selected RL cache membership differs")
    for name, row in caches.items():
        require(row["sha256_matches"] is True and row["compressed_size"] == originals[name]["compressed_size"] and
                row["actual_compressed_sha256"] == row["extraction_receipt_compressed_sha256"] == originals[name]["compressed_sha256"],
                "Selected compressed cache receipt differs: " + name)
    mapped, gap_bytes, selected_bytes = defaultdict(list), 0, 0
    for part in proof["parts"]:
        published = lookup[part["path"]]
        require(part["size"] == published["size"] and part["logical_start"] == published["logical_start"] and
                part["sha256_matches"] is True and part["actual_sha256"] == part["expected_sha256"] == published["sha256"],
                "Recorded covering volume digest differs from fixed publisher lock: " + part["path"])
        segments = []
        for row in part["selected_cache_mappings"]:
            require(row["logical_start"] == part["logical_start"] + row["part_offset"] and
                    row["logical_start"] == caches[row["path"]]["logical_start"] + row["cache_offset"],
                    "Selected cache volume offset differs")
            segments.append((row["part_offset"], row["bytes"]))
            mapped[row["path"]].append((row["cache_offset"], row["bytes"]))
            selected_bytes += row["bytes"]
        for row in part["remote_gap_ranges"]:
            require(row["logical_start"] == part["logical_start"] + row["part_offset"] and
                    bool(re.fullmatch("[0-9a-f]{64}", row["sha256"])), "Remote gap offset/hash differs")
            segments.append((row["part_offset"], row["bytes"]))
            gap_bytes += row["bytes"]
        end = 0
        for start, length in sorted(segments):
            require(start == end and length > 0, "Volume proof has overlap/gap")
            end += length
        require(end == part["size"], "Volume proof does not cover full part")
    for name, segments in mapped.items():
        end = 0
        for start, length in sorted(segments):
            require(start == end, "Compressed cache proof has overlap/gap")
            end += length
        require(end == caches[name]["compressed_size"], "Compressed cache not fully bound")
    require(set(mapped) == set(caches) and gap_bytes == proof["gap_network_bytes"] == proof["expected_gap_network_bytes"] and
            selected_bytes == proof["cached_bytes_hashed"] == plan["compressed_payload_bytes"] and
            proof["selected_RL_payloads_cryptographically_bound_to_published_volume_SHA256"] is True,
            "Volume proof totals/result differ")

    return {"passed": True, "scope": "Independent local audit of recorded actual BW development execution; no fresh remote probe, weight reload or official evaluation",
            "run_id": RUN_ID, "smoke_client_run_id": smoke_id, "client_id_is_separate_from_owner_by_design": smoke_id != RUN_ID,
            "source_files_bound_to_frozen002": sources, "runtime_device": {"gpu_available": env["gpu_available"], "devices": env["devices"],
                "HIP": env["hip"], "HCU_plugin_activated": True, "packages": env["packages"], "hy_smi": env["hy-smi"]},
            "weight_loading": {"observed_DefaultLoader_completed": True, "observed_shard_groups": shard_groups,
                "recorded_weight_load_seconds": 3.52, "nonquantized_strict_loader_contract_in_frozen_source": True,
                "runtime_loader_native_file_sha_captured": False,
                "scope": "Full-parameter load inferred from bound adapter mapping, native nonquantized DefaultLoader completion and subsequent inference; no fresh parameter-by-parameter device enumeration"},
            "HTTP_checks": cases, "functional_checks_passed": True, "independent_cleanup": cleanup,
            "state_stopped_and_owner_bound": True, "volume_receipt_audit": {
                "recorded_covering_volumes_match_published_lock": 14, "selected_RL_payloads_bound": 16,
                "recorded_selected_cache_bytes_hashed": selected_bytes, "recorded_gap_bytes_hashed": gap_bytes,
                "all_26_full_volume_bytes_verified": False, "local_large_volume_bytes_rehashed": False,
                "receipt_unchanged": True, "volume_tools_archive": volume_identity,
                "scope": "Receipt consistency, exact fixed-source identity, publisher digest, mapping and complete interval coverage audit; original large volume/cache bytes are remote"},
            "warnings": {"tokenizer_incorrect_regex": "incorrect regex pattern" in server,
                "ROCm_GELU_tanh_falls_back_to_exact": "Falling back to GELU(approximate='none')" in server,
                "distributed_process_group_not_explicitly_destroyed": "destroy_process_group() was not called" in server,
                "DTK_version_file_missing": env["cat /opt/dtk/.info/version"]["code"] != 0},
            "semantic_quality_verified": False, "same_input_reference_comparison_completed": False,
            "contest_alias_equivalence_verified": False, "official_accuracy_verified": False,
            "official_performance_verified": False, "official_acceptance": False, "submitted": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=ROOT / "dist/000-session007-mobile-pilot001.zip")
    parser.add_argument("--extract", type=Path, default=ROOT / "evidence/session007/mobile-pilot001")
    parser.add_argument("--output", type=Path, default=ROOT / "evidence/session007/mobile-pilot001-independent-audit.json")
    args = parser.parse_args()
    raw, identity = checked_zip(args.archive, RAW_SHA)
    manifest = manifest_check(raw)
    require(len(raw) == 19 and len(manifest) == 18, "Unexpected raw member count")
    if args.extract.exists():
        require(files_in(args.extract) == set(raw) and all((args.extract / name).read_bytes() == data for name, data in raw.items()),
                "Existing originals are not exact; refusing to replace them")
    else:
        for parent in args.extract.parents:
            require(not parent.is_symlink() and not getattr(parent, "is_junction", lambda: False)(), "Extraction ancestor is a link")
        args.extract.mkdir(parents=True, exist_ok=False)
        for name, data in raw.items():
            path = args.extract.joinpath(*PurePosixPath(name).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(data)
        require(files_in(args.extract) == set(raw) and all((args.extract / name).read_bytes() == data for name, data in raw.items()),
                "Extraction readback differs")
    tools, tools_identity = checked_zip(ROOT / "dist/session006-mobilevla-tools-002.zip", TOOLS_SHA)
    tool_manifest = json.loads(tools["TOOLS-MANIFEST-006.json"])["files"]
    require(set(tools) == set(tool_manifest) | {"TOOLS-MANIFEST-006.json"}, "Fixed tools membership differs")
    for name, row in tool_manifest.items():
        require(len(tools[name]) == row["bytes"] and sha(tools[name]) == row["sha256"], "Fixed tool member differs")
    receipt = (ROOT / "evidence/session006/auk-actual-originals007/models/mobile-extraction-receipt.json").read_bytes()
    result = audit(raw, tools, receipt)
    result.update(raw_archive=identity, raw_manifest_payloads_verified=len(manifest),
                  extracted_members_verified=len(raw), extracted_to=str(args.extract.resolve()), frozen_tools_archive=tools_identity)
    write_new(args.output, result)
    print(json.dumps({"output": str(args.output), "passed": result["passed"], "run_id": RUN_ID,
                      "raw_members": len(raw), "source_files_bound": len(result["source_files_bound_to_frozen002"]),
                      "HTTP_checks_passed": len(result["HTTP_checks"]), "covering_volumes_bound": 14,
                      "official_accuracy_verified": False, "submitted": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
