#!/usr/bin/env python3
"""Freeze the MobileVLA 006 CPU tools; never includes model weights or caches."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def package(version="001"):
    paths = []
    for name in ("candidates/mobilevla-r1-session006", "tasks/mobilevla-r1-rl-bw-vllm"):
        root = ROOT / name
        def onerror(error):
            raise error
        for directory, dirs, files in os.walk(root, followlinks=False, onerror=onerror):
            for d in list(dirs):
                path = Path(directory) / d
                if path.is_symlink():
                    raise ValueError(f"No directory links in tool bundle: {path}")
                if d == "__pycache__":
                    dirs.remove(d)
            for filename in files:
                path = Path(directory) / filename
                if path.is_symlink():
                    raise ValueError(f"No links in tool bundle: {path}")
                if filename.endswith(".pyc"):
                    continue
                if path.suffix in (".safetensors", ".bin", ".pt", ".pth", ".pkl", ".so", ".pyd"):
                    raise ValueError(f"Unexpected model/binary in tools: {path}")
                paths.append(path)
    paths.extend(ROOT / p for p in (
        "scripts/recon2reason_runtime.py", "scripts/recon2reason_smoke.py",
        "tests/test_mobilevla_session006.py", "scripts/package_mobilevla_tools_006.py",
        "evidence/mobilevla-preflight-20261023/framework-source/vllm-v0.18.1-llava.py",
    ))
    if version == "002":
        paths.extend(ROOT / p for p in (
            "scripts/probe_mobilevla_native_006.py", "scripts/inspect_mobilevla_headers_006.py",
            "scripts/analyze_mobilevla_header_report_006.py",
        ))
    for path in sorted((ROOT / "evidence/session006/mobilevla-cpu").glob("*")):
        if path.is_file() and not path.is_symlink():
            paths.append(path)
    names = [p.relative_to(ROOT).as_posix() for p in paths]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate tool archive member")
    payload = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in paths}
    manifest = {
        "schema": "mobilevla-rgb-rl-vllm-tools-006-v2" if version == "002" else "mobilevla-rgb-rl-vllm-tools-006-v1",
        "package_version": version, "model_weights_included": False,
        "GPU_verified": False, "official_accuracy_verified": False,
        "official_performance_verified": False, "submission_ready": False,
        "files": {k: {"bytes": len(v), "sha256": digest(v)} for k, v in sorted(payload.items())},
    }
    output = ROOT / f"dist/session006-mobilevla-tools-{version}.zip"
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, raw in sorted(payload.items()):
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = ((0o100755 if name.endswith(".sh") else 0o100644) << 16)
            archive.writestr(info, raw, compress_type=zipfile.ZIP_DEFLATED)
        archive.writestr("TOOLS-MANIFEST-006.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Tool ZIP CRC failed")
        if set(archive.namelist()) != set(payload) | {"TOOLS-MANIFEST-006.json"}:
            raise ValueError("Tool archive member set differs")
        for name, raw in payload.items():
            if archive.read(name) != raw:
                raise ValueError("Tool archive readback differs")
    if any(p.read_bytes() != payload[p.relative_to(ROOT).as_posix()] for p in paths):
        raise ValueError("Tool source changed during package")
    record = {"output": str(output), "bytes": output.stat().st_size,
              "sha256": digest(output.read_bytes()), "files": len(payload),
              "manifest_sha256": digest(json.dumps(manifest, indent=2).encode() + b"\n"),
              "CRC_and_all_payload_readback": True, "source_unchanged": True}
    receipt = ROOT / f"evidence/session006/mobilevla-tools-package-{version}.json"
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=("001", "002"), default="001")
    package(parser.parse_args().version)
