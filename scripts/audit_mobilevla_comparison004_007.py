#!/usr/bin/env python3
"""Audit downloaded004 same-input comparison and EOS records without any GPU access."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

from mobilevla_compare_policy006_004 import compare_records, validate_completion

EXPECTED_TOOLS_SHA = "f248a692df9ae69af29dae264e4ef6e659269593b25057b26eb17e18e2ad2336"
EXPECTED_LOCK_SHA = "63e70e6a2fff22b7339e70878ecab85844bbca822574da3038fbe1fc686be414"
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
    expected_steps = ["prepare-derived", "native-tokenizer-probe", "native-start", "native-1", "native-8", "native-eos-1", "native-stop", "after-native-hy-smi", "hf-1", "hf-8", "hf-eos-1", "compare-1", "compare-8", "compare-eos-1", "final-stop"]
    require([step["name"] for step in result["steps"]] == expected_steps and all(step["status"] == "PASS" and step["returncode"] == 0 for step in result["steps"]), "Actual bounded stage membership/status differs")
    require(0 < result["elapsed_seconds"] <= plan["total_timeout_seconds"] <= 1800
            and all(0 < step["elapsed_seconds"] <= step["timeout_seconds"] for step in result["steps"]), "Controller/stage elapsed bounds differ")
    for name, row in result["files"].items():
        require(len(raw(name)) == row["bytes"] and digest(raw(name)) == row["sha256"], "Controller file seal differs: " + name)
    require(plan["source_identity_before"]["status"] == "PASS" and result["source_identity_after"]["status"] == "PASS", "Source identity checks did not pass")
    require(plan["source_identity_before"]["lock_sha256"] == result["source_identity_after"]["lock_sha256"] == EXPECTED_LOCK_SHA,
            "Requires exact004 source lock")
    receipt = read("derived-prepare-receipt.json")
    require(receipt["schema"] == "mobilevla-derived-hf-view-007-eos004-v1" and receipt["adapter_revision"] == "007-eos004", "Derived receipt is not004")
    expected_modules = {PurePosixPath(name).name: row["sha256"] for name, row in tools["files"].items()
                        if name.startswith("candidates/mobilevla-r1-session006/mobilevla_bw006/") and name.endswith(".py")}
    require(len(expected_modules) == 7 and receipt["adapter_source_files"] == expected_modules, "Derived receipt adapter source binding differs")
    require(receipt["source_unchanged"] is True and len(receipt["source_files"]) == 16 and len(receipt["derived_small_files"]) == 8, "Derived source/small-file receipt incomplete")
    previous, previous_id = checked_archive(ROOT / "dist/000-session007-mobile-comparison002.zip")
    require(previous_id["sha256"] == "df35fd6707b67956b573130c71f394f84df57b496b6c6df7bbd870892a233634", "Historical003 raw identity differs")
    old_receipts = [json.loads(raw) for name, raw in previous.items() if name.endswith("/derived-prepare-receipt.json")]
    require(len(old_receipts) == 1 and receipt["source_files"] == old_receipts[0]["source_files"]
            and receipt["extraction_receipt_sha256"] == old_receipts[0]["extraction_receipt_sha256"],
            "004 original16 source/extraction receipt identities differ from verified003 originals")
    mapping = receipt["publisher_eos_mapping"]
    require(mapping == {"publisher_explicit_eos_token_id": 128009, "source_text_config_eos_token_id": 128009,
                        "source_generation_eos_token_id": [128001, 128009],
                        "derived_root_text_and_generation_eos_token_id": 128009, "source_files_modified": False},
            "004 mapping does not preserve pinned source EOS metadata")
    with zipfile.ZipFile(ROOT / "dist/session007-mobilevla-tools-004.zip") as archive:
        generation = json.loads(archive.read("candidates/mobilevla-r1-session006/reference/rl-metadata/llm/generation_config.json"))
    generation["eos_token_id"] = 128009
    generated = (json.dumps(generation, ensure_ascii=False, indent=2) + "\n").encode()
    require(receipt["derived_small_files"]["generation_config.json"] == {"bytes": len(generated), "sha256": digest(generated)},
            "Derived generation EOS file seal differs from independent recomputation")
    config = read("native-config.json")
    command = config["command"]
    require(config["run_id"] == run_id and command[command.index("--generation-config") + 1] == "vllm", "Native generation-config mode differs from004")
    require(config["model_path"] == receipt["derived"] and config["architecture"] == "MobileVLAForConditionalGeneration006", "Native model/architecture identity differs")
    probe = read("native-tokenizer-probe.json")
    require(probe["status"] == "passed_tokenizer_only" and probe["model_weights_loaded"] is False, "Actual native tokenizer probe did not pass")
    require(probe["source_files"]["backend"]["sha256"] == expected_modules["backend.py"], "Native probed backend differs from004")
    require(probe["source_files"]["native_tokens_main"]["sha256"] == tools["files"]["candidates/mobilevla-r1-session006/framework-source/vllm-multimodal-processing-processor.py"]["sha256"], "Native token processing source differs from audited public source")
    eos_probe = probe["EOS_configuration_probe"]
    require(eos_probe["status"] == "passed_publisher_EOS_configuration" and eos_probe["model_weights_loaded"] is False
            and eos_probe["generation_config_mode"] == "vllm" and eos_probe["ordinary_sampling_defaults"] == {}, "Native EOS configuration probe did not pass")
    require(eos_probe["effective_stop_token_ids"] == [128009] and eos_probe["renderer_primary_eos_token_id"] == eos_probe["effective_eos_token_id"] == 128009,
            "Native stop set contains a nonpublisher EOS")
    require(eos_probe["actual_generation_config_special_tokens"]["eos_token_id"] == 128009
            and eos_probe["historical003_source_generation_effective_stop_token_ids"] == [128001, 128009], "Native EOS leak/fix regression differs")
    require(eos_probe["derived_root_and_text_eos"] == [128009, 128009] and set(eos_probe["actual_config_object_eos"].values()) == {128009}
            and eos_probe["source_original_LLM_config_eos"] == 128009 and eos_probe["source_original_generation_config_eos"] == [128001, 128009], "Actual HF/native config objects differ from source contract")
    require(set(eos_probe["native_source_files"]) == {"model_config", "renderer", "input_processor", "sampling_params"}, "Native EOS method source set incomplete")
    for row in eos_probe["native_source_files"].values():
        require(re.fullmatch("[0-9a-f]{64}", row["sha256"]) is not None, "Native EOS method source digest invalid")
    ready = read("native-ready-state.json")
    require(ready["run_id"] == run_id and ready["status"] == "ready", "Native ready state does not belong to this run")
    cleanup_records = {}
    for phase in ("native-stop", "final-stop"):
        verification, state = read(phase + "-verification.json"), read(phase + "-state.json")
        require(verification["status"] == "PASS" and state["run_id"] == verification["run_id"] == run_id, phase + " owner/cleanup failed")
        require(verification["terminal_state_recorded"] is True and state["status"] == "stopped" and result["native_cleanup" if phase == "native-stop" else "final_cleanup"] == verification, phase + " terminal/result state differs")
        require(verification["port_refused_connection"] is True and not verification["owned_process_group_live_members"] and not verification["process_probe_errors"], phase + " retained a live owned process or port")
        for role in ("supervisor", "server"):
            entry = verification["processes"][role]
            require(entry["pid"] == state[role + "_pid"] == ready[role + "_pid"]
                    and entry["expected_start_token"] == state[role + "_token"] == ready[role + "_token"]
                    and not entry["same_owned_process_running"], phase + " process PID/token mismatch")
        cleanup_records[phase] = verification
    require(datetime.fromisoformat(cleanup_records["native-stop"]["captured_utc"]) < datetime.fromisoformat(next(step["started_utc"] for step in result["steps"] if step["name"] == "hf-1")), "HF started before independent native cleanup")
    comparisons = []
    EOS_observed = False
    for case, frames, maximum in (("1", 1, 32), ("8", 8, 32), ("eos-1", 1, 128)):
        native, reference, saved = read(f"native-{case}.json"), read(f"hf-{case}.json"), read(f"comparison-{case}.json")
        policy = native["generation_policy"]
        require(policy["max_new_tokens"] == maximum and policy["eos_token_ids"] == [128009] and policy["mode"] == "greedy", "Comparison sampling policy differs from publisher greedy policy")
        source_ids = native["source_prompt_token_ids"]
        require(source_ids[:2] == [128000, 128000] and source_ids.count(-200) == frames, "Publisher BOS/frame contract differs")
        if case != "eos-1":
            observation = next(row for row in probe["observations"] if row["frames"] == frames)
            require(observation["publisher_ids"] == source_ids and observation["matches_publisher"] is True,
                    "Native-only preflight source IDs differ from actual request reference")
            require(observation["native_source_equivalent_ids"] == [128259 if token == -200 else token for token in source_ids],
                    "Native tokens-only preflight IDs differ from actual request source contract")
        else:
            require(native["question"] == reference["question"] == "请用一句话描述图像中的主要物体。", "Long stop-boundary question differs")
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
        require(saved["input_records"]["native"]["sha256"] == digest(raw(f"native-{case}.json")) and
                saved["input_records"]["reference"]["sha256"] == digest(raw(f"hf-{case}.json")), "Comparison input record seal differs")
        require(expected["comparable"] is True, "Input contracts not comparable")
        EOS_observed |= (native["generated_token_ids"][-1] == 128009 and native["finish_reason"] == "stop" and
                         reference["generated_token_ids"][-1] == 128009 and reference["finish_reason"] == "stop")
        comparisons.append({"case": case, "frames": frames, "expanded_input_tokens": len(expanded), "raw_generated_tokens": len(native["generated_token_ids"]),
                            "native_finish_reason": native["finish_reason"], "HF_finish_reason": reference["finish_reason"],
                            "raw_generated_token_ids_equal": expected["raw_generated_token_ids_equal"], "first_divergence": expected["first_generated_divergence"],
                            "native_stop_boundary_coverage": expected["native_stop_boundary_coverage"],
                            "reference_stop_boundary_coverage": expected["reference_stop_boundary_coverage"]})
    equal = all(row["raw_generated_token_ids_equal"] for row in comparisons)
    exercised = all(comparisons[-1][label + "_stop_boundary_coverage"]["stop_boundary_exercised"] is True for label in ("native", "reference"))
    expected_status = "completed_equal_raw_tokens" if equal and exercised else ("completed_equal_without_stop_boundary_coverage" if equal else "completed_with_divergence_or_incomparable_records")
    require(result["status"] == expected_status and result["publisher_stop_boundary_exercised"] == exercised, "Controller completion status differs from raw evidence")
    return {"schema": "mobilevla-comparison007-004-independent-audit-v1", "audit_integrity_passed": True,
            "same_input_1_and_8_frame_comparison_completed": True, "raw_generated_tokens_equal": equal,
            "run_id": run_id, "controller_elapsed_seconds": result["elapsed_seconds"], "comparisons": comparisons,
            "source_lock_sha256": EXPECTED_LOCK_SHA, "derived_adapter_source_modules_bound": len(expected_modules),
            "original16_source_and_extraction_receipt_SHA_match_verified003": True,
            "actual_native_source_files": probe["source_files"], "actual_tokenizer_classes": probe["tokenizer_classes"],
            "actual_native_package_versions": probe["package_versions"],
            "new_derived_receipt_schema": receipt["schema"], "cleanup_records": cleanup_records,
            "EOS_runtime_trigger_observed": EOS_observed, "publisher_stop_boundary_exercised": exercised,
            "actual_native_EOS_configuration_probe": eos_probe, "navigation_accuracy_verified": False,
            "official_accuracy_verified": False, "official_performance_verified": False, "official_acceptance": False,
            "submitted": False, "scope": "Downloaded actual BW records audited locally; no new GPU execution, no local rehash of remote17GB weights, no official evaluator"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--prefix")
    parser.add_argument("--tools", type=Path, default=ROOT / "dist/session007-mobilevla-tools-004.zip")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Audit output must be new")
    payload, identity = checked_archive(args.archive)
    prefixes = [name.rsplit("/", 1)[0] for name in payload if name.endswith("/orchestrator-result.json")]
    require(bool(args.prefix) or len(prefixes) == 1, "Select a unique comparison prefix")
    prefix = args.prefix or prefixes[0]
    require(digest(args.tools.read_bytes()) == EXPECTED_TOOLS_SHA, "Expected exact frozen004 tools archive")
    with zipfile.ZipFile(args.tools) as archive:
        tools = json.loads(archive.read("TOOLS-MANIFEST-007-004.json"))
    record = audit(payload, prefix, tools)
    record["raw_archive"] = identity
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0 if record["raw_generated_tokens_equal"] and record["publisher_stop_boundary_exercised"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
