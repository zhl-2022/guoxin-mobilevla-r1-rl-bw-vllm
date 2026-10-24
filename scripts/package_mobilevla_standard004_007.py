#!/usr/bin/env python3
"""Seal a standard source/evidence deliverable only after actual004 comparison and standard pilot evidence pass."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser()
    default_stage = ROOT / "deliverables/mobilevla-session007-004-staging"
    parser.add_argument("--stage", type=Path, default=default_stage if default_stage.is_dir() else ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/session007-mobilevla-standard-004.zip")
    parser.add_argument("--receipt", type=Path, default=ROOT / "evidence/session007/mobilevla-standard004-package.json")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    stage = args.stage.resolve(strict=True)
    state_file = stage / "tasks/mobilevla-r1-rl-bw-vllm-standard007/delivery-state.json"
    if not state_file.is_file():
        state_file = stage / "tasks/mobilevla-r1-rl-bw-vllm-delivery007-draft/delivery-state.json"
    state = json.loads(state_file.read_text(encoding="utf-8"))
    gaps = [key for key in ("same_input_reference_completed", "comparison_1_and_8_raw_tokens_equal", "native_task_GPU_replayed", "eos_policy_configuration_verified", "publisher_stop_boundary_exercised") if state.get(key) is not True]
    report = (stage / "README.md").read_text(encoding="utf-8")
    headings = [line for line in report.splitlines() if line.startswith("## ")]
    if len(headings) != 8 or any(not line.startswith(f"## {index}. ") for index, line in enumerate(headings, 1)):
        gaps.append("eight_section_report")
    if any(marker in report for marker in ("原件待补入", "原件待纳入", "尚待纳入", "等待完整重跑")):
        gaps.append("report_still_has_pending_runtime_evidence")
    if args.plan:
        print(json.dumps({"plan_only": True, "stage": str(stage), "ready_to_seal": not gaps, "gaps": gaps,
                          "official_accuracy_verified": False, "official_performance_verified": False, "submitted": False}, indent=2))
        return 0 if not gaps else 2
    if gaps:
        raise ValueError("Cannot seal while required actual004 evidence is missing: " + ", ".join(gaps))
    if args.output.exists() or args.receipt.exists():
        raise FileExistsError("Standard ZIP and package receipt must both be new")
    import sys
    sys.path.insert(0, str(stage / "scripts"))
    import mobilevla_comparison_orchestrator007_003 as controller
    source_check = controller.check_sources(stage, stage)
    import audit_mobilevla_comparison004_007 as comparison_auditor
    import audit_mobilevla_pilot004_007 as pilot_auditor
    def sealed_raw(key):
        path = stage / state[key]
        if not path.resolve(strict=True).is_relative_to(stage):
            raise ValueError("Actual evidence archive must belong to this final source tree")
        return comparison_auditor.checked_archive(path)[0]
    tools_path = stage / "dist/session007-mobilevla-tools-004.zip"
    if digest(tools_path.read_bytes()) != comparison_auditor.EXPECTED_TOOLS_SHA:
        raise ValueError("Final tree does not include exact frozen004 deployment tools")
    with zipfile.ZipFile(tools_path) as archive:
        tool_payload = {name: archive.read(name) for name in archive.namelist()}
        tools = json.loads(tool_payload["TOOLS-MANIFEST-007-004.json"])
    comparison_audit = comparison_auditor.audit(sealed_raw("comparison_raw_archive"), state["comparison_raw_prefix"], tools)
    pilot_audit = pilot_auditor.audit(sealed_raw("standard_pilot_raw_archive"), state["standard_pilot_raw_prefix"], state["standard_pilot_cleanup_member"], tool_payload)
    if not (comparison_audit["raw_generated_tokens_equal"] and comparison_audit["publisher_stop_boundary_exercised"]
            and pilot_audit["functional_checks_passed"]):
        raise ValueError("Final raw comparison/pilot recomputation failed")
    if state["publisher_terminal_EOS_observed"] != comparison_audit["EOS_runtime_trigger_observed"]:
        raise ValueError("Final state terminal EOS claim differs from actual raw evidence")
    crossed = all(comparison_audit["comparisons"][-1][label + "_stop_boundary_coverage"]["continued_after_nonpublisher128001"]
                  for label in ("native", "reference"))
    if state["nonpublisher128001_crossed"] != crossed:
        raise ValueError("Final state128001 crossing claim differs from actual raw evidence")
    payload = {}
    def onerror(error):
        raise error
    for directory, dirs, files in os.walk(stage, followlinks=False, onerror=onerror):
        for name in list(dirs):
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError("No linked directory in standard source package")
            if name in {".git", "__pycache__", ".venv", "venv", "runtime", "pilot-evidence"}:
                dirs.remove(name)
        for name in files:
            path = Path(directory) / name
            if path == stage / "STANDARD-MANIFEST-007-004.json":
                continue
            if path.is_symlink():
                raise ValueError("No linked file in standard source package")
            if path.suffix == ".pyc":
                continue
            if path.suffix in {".safetensors", ".bin", ".pt", ".pth", ".pkl", ".so", ".pyd"}:
                raise ValueError("No weight/cache/native binary in standard source package: " + str(path))
            if path.stat().st_size > 20 * 1024 * 1024:
                raise ValueError("Unexpected large source/evidence artifact")
            payload[path.relative_to(stage).as_posix()] = path.read_bytes()
    if sum(map(len, payload.values())) > 64 * 1024 * 1024:
        raise ValueError("Source package exceeds expected transport budget")
    manifest = {"schema": "mobilevla-standard-development-delivery007-004-v1", "adapter_revision": "007-eos004",
                "development_evidence_sealed": True, "model_weights_included": False, "source_identity": source_check,
                "same_input_reference_completed": True, "native_task_GPU_replayed": True,
                "standard_wrapper_GPU_replayed": False,
                "EOS_comparison_runtime_trigger_observed": state.get("publisher_terminal_EOS_observed") is True,
                "comparison_crossed_nonpublisher128001": state.get("nonpublisher128001_crossed") is True,
                "publisher_stop_boundary_exercised": True, "official_accuracy_verified": False,
                "official_performance_verified": False, "official_acceptance": False, "submission_ready": False, "submitted": False,
                "files": {name: {"bytes": len(raw), "sha256": digest(raw)} for name, raw in sorted(payload.items())}}
    manifest_raw = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, raw in sorted({**payload, "STANDARD-MANIFEST-007-004.json": manifest_raw}.items()):
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = ((0o100755 if name.endswith(".sh") else 0o100644) << 16)
            archive.writestr(info, raw, compress_type=zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None or len(archive.infolist()) != len(payload) + 1:
            raise ValueError("Standard ZIP CRC/member validation failed")
        for name, raw in payload.items():
            if archive.read(name) != raw:
                raise ValueError("Standard ZIP payload readback differs: " + name)
    if any((stage / name).read_bytes() != raw for name, raw in payload.items()):
        raise RuntimeError("Stage changed while packaging")
    record = {"zip": str(args.output.resolve()), "bytes": args.output.stat().st_size,
              "sha256": digest(args.output.read_bytes()), "manifest_sha256": digest(manifest_raw),
              "payload_files": len(payload), "archive_members": len(payload) + 1,
              "CRC_and_all_payload_readback": True, "stage_unchanged": True,
              "development_evidence_sealed": True, "official_accuracy_verified": False,
              "official_performance_verified": False, "submission_ready": False, "submitted": False}
    with args.receipt.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
