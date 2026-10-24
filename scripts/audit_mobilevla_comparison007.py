#!/usr/bin/env python3
"""Audit downloaded 003 same-input comparison records without any GPU access."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

from mobilevla_compare_policy006_003 import compare_records, validate_completion

EXPECTED_TOOLS_SHA = "aa62f6e5dde19cd5fcd4ed6c1cfb599a8a7739d32b9a58bbd8e1fe83522018e6"
EXPECTED_LOCK_SHA = "f27db0276bc887eeb0c14f5e9deff9af40657017c2caa00cb07d596ac2ef33b9"
ROOT = Path(__file__).resolve().parents[1]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_archive(path):
    raw = Path(path).read_bytes()
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        require(len(infos) == len({i.filename for i in infos}), "Duplicate raw archive members")
        require(sum(i.file_size for i in infos) <= 128 * 1024 * 1024, "Raw archive exceeds audit bound")
        require(archive.testzip() is None, "Raw archive CRC failed")
        payload = {}
        for info in infos:
            pure = PurePosixPath(info.filename)
            require(not pure.is_absolute() and ".." not in pure.parts and "\\" not in info.filename and ":" not in info.filename,
                    "Unsafe raw archive path")
            require(not info.is_dir(), "Expected an archive of raw regular files")
            payload[info.filename] = archive.read(info)
    manifest = json.loads(payload["MANIFEST.json"])["files"]
    require(set(payload) == set(manifest) | {"MANIFEST.json"}, "Raw manifest member set differs")
    for name, row in manifest.items():
        require(len(payload[name]) == row["bytes"] and digest(payload[name]) == row["sha256"], "Raw manifest digest differs: " + name)
    return payload, {"path": str(Path(path).resolve()), "bytes": len(raw), "sha256": digest(raw), "verified_payload_files": len(manifest)}


def audit(payload, prefix, tools):
    def raw(name):
        return payload[prefix + "/" + name]
    def read(name):
        return json.loads(raw(name))
    result = read("orchestrator-result.json")
    plan = read("orchestrator-plan.json")
    run_id = result["run_id"]
    require(re.fullmatch("[0-9a-f]{32}", run_id) is not None and plan["run_id"] == run_id, "Controller run identity differs")
    for name, row in result["files"].items():
        require(len(raw(name)) == row["bytes"] and digest(raw(name)) == row["sha256"], "Controller file seal differs: " + name)
    require(plan["source_identity_before"]["status"] == "PASS" and result["source_identity_after"]["status"] == "PASS", "Source identity checks did not pass")
    require(plan["source_identity_before"]["lock_sha256"] == result["source_identity_after"]["lock_sha256"] == EXPECTED_LOCK_SHA,
            "Requires exact003 source lock")
    receipt = read("derived-prepare-receipt.json")
    require(receipt["schema"] == "mobilevla-derived-hf-view-007-bos003-v1" and receipt["adapter_revision"] == "007-bos003", "Derived receipt is not003")
    expected_modules = {PurePosixPath(name).name: row["sha256"] for name, row in tools["files"].items()
                        if name.startswith("candidates/mobilevla-r1-session006/mobilevla_bw006/") and name.endswith(".py")}
    require(len(expected_modules) == 7 and receipt["adapter_source_files"] == expected_modules, "Derived receipt adapter source binding differs")
    require(receipt["source_unchanged"] is True and len(receipt["source_files"]) == 16 and len(receipt["derived_small_files"]) == 8, "Derived source/small-file receipt incomplete")
    config = read("native-config.json")
    command = config["command"]
    require(config["run_id"] == run_id and command[command.index("--generation-config") + 1] == "vllm", "Native command did not ignore non-publisher generation defaults")
    require(config["model_path"] == receipt["derived"] and config["architecture"] == "MobileVLAForConditionalGeneration006", "Native model/architecture identity differs")
    probe = read("native-tokenizer-probe.json")
    require(probe["status"] == "passed_tokenizer_only" and probe["model_weights_loaded"] is False, "Actual native tokenizer probe did not pass")
    require(probe["source_files"]["backend"]["sha256"] == expected_modules["backend.py"], "Native probed backend differs from003")
    require(probe["source_files"]["native_tokens_main"]["sha256"] == tools["files"]["candidates/mobilevla-r1-session006/framework-source/vllm-multimodal-processing-processor.py"]["sha256"], "Native token processing source differs from audited public source")
    ready = read("native-ready-state.json")
    require(ready["run_id"] == run_id and ready["status"] == "ready", "Native ready state does not belong to this run")
    cleanup_records = {}
    for phase in ("native-stop", "final-stop"):
        verification, state = read(phase + "-verification.json"), read(phase + "-state.json")
        require(verification["status"] == "PASS" and state["run_id"] == verification["run_id"] == run_id, phase + " owner/cleanup failed")
        require(verification["port_refused_connection"] is True and not verification["owned_process_group_live_members"] and not verification["process_probe_errors"], phase + " retained a live owned process or port")
        for role in ("supervisor", "server"):
            entry = verification["processes"][role]
            require(entry["expected_start_token"] == ready[role + "_token"] and not entry["same_owned_process_running"], phase + " process token mismatch")
        cleanup_records[phase] = verification
    comparisons = []
    EOS_observed = False
    for frames in (1, 8):
        native, reference, saved = read(f"native-{frames}.json"), read(f"hf-{frames}.json"), read(f"comparison-{frames}.json")
        policy = native["generation_policy"]
        require(policy["max_new_tokens"] == 32 and policy["eos_token_ids"] == [128009] and policy["mode"] == "greedy", "Comparison sampling policy differs from publisher greedy policy")
        source_ids = native["source_prompt_token_ids"]
        require(source_ids[:2] == [128000, 128000] and source_ids.count(-200) == frames, "Publisher BOS/frame contract differs")
        observation = next(row for row in probe["observations"] if row["frames"] == frames)
        require(observation["publisher_ids"] == source_ids and observation["matches_publisher"] is True,
                "Native-only preflight source IDs differ from actual request reference")
        require(observation["native_source_equivalent_ids"] == [128259 if token == -200 else token for token in source_ids],
                "Native tokens-only preflight IDs differ from actual request source contract")
        expanded = []
        for token in source_ids:
            expanded.extend([128259] * 196 if token == -200 else [token])
        require(native["actual_prompt_token_ids"] == native["expected_expanded_prompt_token_ids"] == expanded, "Native expanded input differs")
        response = native["response"]
        choice = response["choices"][0]
        require(response["prompt_token_ids"] == expanded and choice["token_ids"] == native["generated_token_ids"], "Raw API IDs differ from saved normalized fields")
        require(not validate_completion(choice["token_ids"], choice["finish_reason"], choice.get("stop_reason"), policy, response["usage"]), "Raw API generation accounting failed")
        usage = response["usage"]
        require(usage["prompt_tokens"] == len(expanded) and usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], "Raw API total/prompt usage differs")
        for key, expected in (("pixel_shape", [frames, 3, 384, 384]), ("vision_shape", [frames, 729, 1152]),
                              ("visual_embedding_shape", [frames, 196, 4096]), ("expanded_input_embedding_shape", [1, len(expanded), 4096])):
            require(reference[key] == expected, "HF reference tensor shape differs: " + key)
        for key in ("pixel_tensor_sha256", "vision_tensor_sha256", "visual_embedding_sha256"):
            require(re.fullmatch("[0-9a-f]{64}", reference[key]) is not None, "HF tensor digest missing: " + key)
        expected = compare_records(native, reference)
        for key, value in expected.items():
            require(saved.get(key) == value, "Saved comparison differs from independent recomputation: " + key)
        require(saved["input_records"]["native"]["sha256"] == digest(raw(f"native-{frames}.json")) and
                saved["input_records"]["reference"]["sha256"] == digest(raw(f"hf-{frames}.json")), "Comparison input record seal differs")
        require(expected["comparable"] is True, "Input contracts not comparable")
        EOS_observed |= native["finish_reason"] == "stop" or reference["finish_reason"] == "stop"
        comparisons.append({"frames": frames, "expanded_input_tokens": len(expanded), "raw_generated_tokens": len(native["generated_token_ids"]),
                            "native_finish_reason": native["finish_reason"], "HF_finish_reason": reference["finish_reason"],
                            "raw_generated_token_ids_equal": expected["raw_generated_token_ids_equal"], "first_divergence": expected["first_generated_divergence"]})
    equal = all(row["raw_generated_token_ids_equal"] for row in comparisons)
    require(result["status"] == ("completed_equal_raw_tokens" if equal else "completed_with_divergence_or_incomparable_records"), "Controller completion status differs from raw evidence")
    return {"schema": "mobilevla-comparison007-003-independent-audit-v1", "audit_integrity_passed": True,
            "same_input_1_and_8_frame_comparison_completed": True, "raw_generated_tokens_equal": equal,
            "run_id": run_id, "controller_elapsed_seconds": result["elapsed_seconds"], "comparisons": comparisons,
            "source_lock_sha256": EXPECTED_LOCK_SHA, "derived_adapter_source_modules_bound": len(expected_modules),
            "actual_native_source_files": probe["source_files"], "actual_tokenizer_classes": probe["tokenizer_classes"],
            "actual_native_package_versions": probe["package_versions"],
            "new_derived_receipt_schema": receipt["schema"], "cleanup_records": cleanup_records,
            "EOS_runtime_trigger_observed": EOS_observed, "navigation_accuracy_verified": False,
            "official_accuracy_verified": False, "official_performance_verified": False, "official_acceptance": False,
            "submitted": False, "scope": "Downloaded actual BW records audited locally; no new GPU execution, no local rehash of remote17GB weights, no official evaluator"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--prefix")
    parser.add_argument("--tools", type=Path, default=ROOT / "dist/session007-mobilevla-tools-003.zip")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Audit output must be new")
    payload, identity = checked_archive(args.archive)
    prefixes = [name.rsplit("/", 1)[0] for name in payload if name.endswith("/orchestrator-result.json")]
    require(bool(args.prefix) or len(prefixes) == 1, "Select a unique comparison prefix")
    prefix = args.prefix or prefixes[0]
    require(digest(args.tools.read_bytes()) == EXPECTED_TOOLS_SHA, "Expected exact frozen003 tools archive")
    with zipfile.ZipFile(args.tools) as archive:
        tools = json.loads(archive.read("TOOLS-MANIFEST-007-003.json"))
    record = audit(payload, prefix, tools)
    record["raw_archive"] = identity
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0 if record["raw_generated_tokens_equal"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
