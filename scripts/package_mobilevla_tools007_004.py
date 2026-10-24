#!/usr/bin/env python3
"""Seal the independent004 deployment tools after local validation only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "revisions/mobilevla-eos004")
    parser.add_argument("--output", type=Path, default=ROOT / "dist/session007-mobilevla-tools-004.zip")
    parser.add_argument("--receipt", type=Path, default=ROOT / "evidence/session007/mobilevla-tools004-package.json")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    spec = importlib.util.spec_from_file_location("mobile004orchestrator", root / "scripts/mobilevla_comparison_orchestrator007_003.py")
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    source_check = controller.check_sources(root, root)
    payload = {}
    def onerror(error):
        raise error
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=onerror):
        for name in list(dirs):
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError("Linked tool directories are forbidden")
            if name in {"__pycache__", ".git", "runtime"}:
                dirs.remove(name)
        for name in files:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError("Linked tool files are forbidden")
            if path.suffix in {".pyc", ".safetensors", ".pt", ".pth", ".bin", ".pkl", ".so", ".pyd"}:
                raise ValueError("Unexpected model/cache/native binary in tools")
            if path.stat().st_size > 12 * 1024 * 1024:
                raise ValueError("Unexpected large deployment tool member")
            payload[path.relative_to(root).as_posix()] = path.read_bytes()
    manifest = {"schema": "mobilevla-rgb-rl-vllm-tools007-eos004-v1", "package_version": "004",
                "adapter_revision": "007-eos004", "parent003_zip_sha256": "aa62f6e5dde19cd5fcd4ed6c1cfb599a8a7739d32b9a58bbd8e1fe83522018e6",
                "source_identity": source_check, "CPU_tests_passed": 45, "GPU004_verified": False,
                "official_accuracy_verified": False, "official_performance_verified": False,
                "submitted": False, "model_weights_included": False,
                "files": {name: {"bytes": len(raw), "sha256": sha(raw)} for name, raw in sorted(payload.items())}}
    manifest_raw = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    member = "TOOLS-MANIFEST-007-004.json"
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, raw in sorted({**payload, member: manifest_raw}.items()):
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = ((0o100755 if name.endswith(".sh") else 0o100644) << 16)
            archive.writestr(info, raw, compress_type=zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(payload) | {member}:
            raise ValueError("004 CRC/member validation failed")
        for name, raw in payload.items():
            if archive.read(name) != raw or (root / name).read_bytes() != raw:
                raise ValueError("004 readback/source identity differs")
    record = {"output": str(args.output.resolve()), "bytes": args.output.stat().st_size,
              "sha256": sha(args.output.read_bytes()), "manifest_sha256": sha(manifest_raw),
              "payload_files": len(payload), "archive_members": len(payload) + 1,
              "source_identity": source_check, "CRC_and_all_payload_readback": True,
              "source_unchanged": True, "GPU004_verified": False, "submitted": False}
    with args.receipt.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
