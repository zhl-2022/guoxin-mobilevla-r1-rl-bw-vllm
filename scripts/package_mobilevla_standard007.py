#!/usr/bin/env python3
"""Seal a standard source/evidence deliverable only after both003 runs pass."""
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
    parser.add_argument("--stage", type=Path, default=ROOT / "deliverables/mobilevla-session007-003-staging")
    parser.add_argument("--output", type=Path, default=ROOT / "dist/session007-mobilevla-standard-003.zip")
    parser.add_argument("--receipt", type=Path, default=ROOT / "evidence/session007/mobilevla-standard003-package.json")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    stage = args.stage.resolve(strict=True)
    state = json.loads((stage / "tasks/mobilevla-r1-rl-bw-vllm-delivery007-draft/delivery-state.json").read_text(encoding="utf-8"))
    gaps = [key for key in ("same_input_reference_completed", "comparison_1_and_8_raw_tokens_equal", "native_standard_task_GPU_replayed") if state.get(key) is not True]
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
        raise ValueError("Cannot seal while required actual003 evidence is missing: " + ", ".join(gaps))
    if args.output.exists() or args.receipt.exists():
        raise FileExistsError("Standard ZIP and package receipt must both be new")
    import sys
    sys.path.insert(0, str(stage / "scripts"))
    import mobilevla_comparison_orchestrator007_002 as controller
    source_check = controller.check_sources(stage, stage)
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
    manifest = {"schema": "mobilevla-standard-development-delivery007-003-v1", "adapter_revision": "007-bos003",
                "development_evidence_sealed": True, "model_weights_included": False, "source_identity": source_check,
                "same_input_reference_completed": True, "standard_native_GPU_replayed": True,
                "EOS_comparison_runtime_trigger_observed": False, "official_accuracy_verified": False,
                "official_performance_verified": False, "official_acceptance": False, "submission_ready": False, "submitted": False,
                "files": {name: {"bytes": len(raw), "sha256": digest(raw)} for name, raw in sorted(payload.items())}}
    manifest_raw = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, raw in sorted(payload.items()):
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = ((0o100755 if name.endswith(".sh") else 0o100644) << 16)
            archive.writestr(info, raw, compress_type=zipfile.ZIP_DEFLATED)
        archive.writestr("STANDARD-MANIFEST-007-003.json", manifest_raw)
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
