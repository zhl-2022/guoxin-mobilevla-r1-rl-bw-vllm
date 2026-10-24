"""Seal a new AuK bundle from immutable004 and locally audited remote originals.

Copyright scope notice: no new repository-wide open-source license selected.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

from verify_auk_final007_bundle import files_in, verify

ROOT = Path(__file__).resolve().parents[1]
BASE_SHA = "b9cbb0f9989b59cc10bfe9b6b9686534e1f8ba8a96baf25219659feb7facb5b7"
SOURCE_SHA = "f087c025de08e7f764118d44f187ff7de78e9ca265539a89d353b18b95c0bdfb"
PREVIOUS_SHA = "169cccceadf0d26840251bb7c093c1883842072402da49ee2939b44e63004f42"
REMOTE_SHA = "0b7172198eb446bf875457c3441f089e7715f800e19153aaaf21b002c09e98ba"
RUN_ID = "a3e59176c8456dbbefd89b2189cb4165"
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def document(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def checked_zip(path, expected_sha=None):
    raw = path.read_bytes()
    if expected_sha and sha256(raw) != expected_sha:
        raise ValueError("Frozen ZIP identity differs: " + str(path))
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC failure: " + str(path))
        result = {}
        for info in archive.infolist():
            name = info.filename
            rel = PurePosixPath(name)
            if (rel.is_absolute() or ".." in rel.parts or "\\" in name
                    or any(":" in part for part in rel.parts)
                    or (not info.is_dir() and rel.as_posix() != name)
                    or (info.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError("Unsafe archive member: " + name)
            if info.is_dir():
                continue
            if name in result or info.file_size > 64 * 1024 * 1024:
                raise ValueError("Duplicate or oversized archive member: " + name)
            if rel.suffix.lower() in WEIGHT_SUFFIXES or any(
                part in {".venv", "venv", "venv002", "__pycache__", ".cache", ".git"}
                for part in rel.parts
            ):
                raise ValueError("Model/environment/cache material is not a delivery member: " + name)
            result[name] = archive.read(info)
    return result, {"path": path.name, "bytes": len(raw), "sha256": sha256(raw), "files": len(result)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-archive", type=Path, required=True)
    parser.add_argument("--pilot-audit", type=Path, required=True)
    parser.add_argument("--failure-audit", type=Path, required=True)
    parser.add_argument("--extraction-receipt", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=ROOT / "tasks/auk-bw-sgomni-final007")
    parser.add_argument("--output", type=Path, default=ROOT / "dist/session006-auk-standard-final007.zip")
    args = parser.parse_args()
    if args.destination.exists() or args.output.exists():
        raise ValueError("Refusing to overwrite any final delivery")

    payload, base = checked_zip(ROOT / "dist/session006-auk-standard-delivery-004.zip", BASE_SHA)
    frozen, source = checked_zip(ROOT / "dist/session006-auk-port-cpu-004.zip", SOURCE_SHA)
    if len(payload) != 352 or len(frozen) != 326:
        raise ValueError("Unexpected immutable004 archive membership")
    for name, raw in frozen.items():
        if payload["candidate/" + name] != raw:
            raise ValueError("Standard004 candidate differs from frozen004: " + name)
    remote, remote_identity = checked_zip(args.remote_archive, REMOTE_SHA)
    original_manifest = json.loads(remote["MANIFEST.json"])
    if set(remote) != set(original_manifest) | {"MANIFEST.json"}:
        raise ValueError("Original raw manifest membership differs")
    for name, row in original_manifest.items():
        raw = remote[name]
        if len(raw) != row["size"] or sha256(raw) != row["sha256"]:
            raise ValueError("Original raw manifest identity differs: " + name)
    audit = json.loads(args.pilot_audit.read_text(encoding="utf-8"))
    if not audit["passed"] or audit["run_id"] != RUN_ID:
        raise ValueError("A successful local audit for the actual pilot004 is required")
    # Bind the audit's every consumed raw byte to the preserved remote archive.
    prefixes = [name[:-len("pilot-result.json")] for name in remote
                if name.endswith("/" + RUN_ID + "/pilot-result.json")]
    if len(prefixes) != 1:
        raise ValueError("Remote archive must contain exactly one actual pilot004")
    for name, row in audit["raw_files"].items():
        raw = remote[prefixes[0] + name]
        if len(raw) != row["bytes"] or sha256(raw) != row["sha256"]:
            raise ValueError("Audited raw bytes differ from original ZIP: " + name)

    payload["historical/standard004-README.md"] = payload.pop("README.md")
    payload["README.md"] = args.report.read_bytes()
    payload["evidence/pilot004-local-audit.json"] = args.pilot_audit.read_bytes()
    failure = json.loads(args.failure_audit.read_text(encoding="utf-8"))
    if not failure["passed"] or failure["pilot_passed"] is not False or failure["frozen003_sha256"] != PREVIOUS_SHA:
        raise ValueError("Expected-failure audit does not preserve pilot003 status")
    for name, row in failure["raw_files"].items():
        raw = remote["auk003/" + failure["run_id"] + "/" + name]
        if len(raw) != row["bytes"] or sha256(raw) != row["sha256"]:
            raise ValueError("Failed pilot raw bytes differ: " + name)
    receipt = json.loads(args.extraction_receipt.read_text(encoding="utf-8"))
    if not receipt["passed"] or receipt["archive"]["sha256"] != REMOTE_SHA:
        raise ValueError("Raw extraction receipt differs")
    payload["evidence/pilot003-failure-audit.json"] = args.failure_audit.read_bytes()
    payload["evidence/originals-extraction-receipt.json"] = args.extraction_receipt.read_bytes()
    payload["evidence/remote-originals.zip"] = args.remote_archive.read_bytes()
    for name, raw in remote.items():
        payload["evidence/remote-originals/" + name] = raw
    native = ROOT / "evidence/session006/remote-native-preflights001"
    native_members = files_in(native)
    if len(native_members) != 18:
        raise ValueError("Native preflight original membership changed")
    for name in sorted(native_members):
        payload["evidence/native-preflights001/" + name] = (native / name).read_bytes()
    payload["archives/session006-auk-port-cpu-004.zip"] = (ROOT / "dist/session006-auk-port-cpu-004.zip").read_bytes()
    checked_zip(ROOT / "dist/session006-auk-port-cpu-003.zip", PREVIOUS_SHA)
    payload["archives/session006-auk-port-cpu-003.zip"] = (ROOT / "dist/session006-auk-port-cpu-003.zip").read_bytes()
    model_lock = json.loads(frozen["model-lock.json"])
    for model, namespace in zip(model_lock["models"], ("auk006", "qwen25-omni3b006")):
        row = next(item for item in model["files"] if item["path"] == "LICENSE")
        raw = (ROOT / "evidence/session006/source-mapping/small-files" / namespace / "LICENSE").read_bytes()
        identity = sha256(raw) if row.get("sha256") else hashlib.sha1(
            f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if len(raw) != row["bytes"] or identity != (row.get("sha256") or row["git_blob_sha1"]):
            raise ValueError("Fixed model license identity differs: " + model["id"])
        payload["licenses/" + namespace + "/LICENSE"] = raw
    payload["model-acquisition-lock.json"] = (ROOT / "evidence/session006/source-mapping/auk-model-download-lock006.json").read_bytes()
    for name in ("download_auk_models_006.py", "download_candidate_models_20261023.py", "extract_auk_actual007.py"):
        payload["tools/" + name] = (ROOT / "scripts" / name).read_bytes()
    for name in ("session007-A-B-autooff.png", "A-gpu-autooff-B-cpu-001.png"):
        payload["evidence/platform/" + name] = (ROOT / "evidence/session006/screenshots" / name).read_bytes()
    payload["verify_bundle.py"] = (ROOT / "scripts/verify_auk_final007_bundle.py").read_bytes()
    payload["tools/audit_auk_pilot006_actual.py"] = (ROOT / "scripts/audit_auk_pilot006_actual.py").read_bytes()
    payload["tools/build_auk_final007.py"] = Path(__file__).read_bytes()
    payload["tools/verify_auk_final007_bundle.py"] = (ROOT / "scripts/verify_auk_final007_bundle.py").read_bytes()
    task = json.loads(payload["task.json"])
    task.update(local_actual_pilot_audit_passed=True, actual_run_id=RUN_ID,
                official_acceptance=False, submission_ready=False, submitted=False)
    payload["task.json"] = document(task)
    payload["provenance.json"] = document({
        "base_standard004": base, "frozen_candidate004": source,
        "remote_originals": remote_identity, "actual_pilot_prefix": prefixes[0],
        "base_source_and_cpu_wheels_preserved": True,
        "historical_standard004_readme": "historical/standard004-README.md",
        "standard_wrapper_GPU_rerun": False,
        "audit_script_command": "python -B tools/audit_auk_pilot006_actual.py --pilot-dir evidence/remote-originals/" + prefixes[0] + " --source-archive archives/session006-auk-port-cpu-004.zip --protocol candidate/protocol.py --output new-pilot004-audit.json",
        "scope": "Development reproduction and actual004 evidence; final competition submission is user-owned",
    })
    manifest = {"schema": "auk-final007-delivery-v1", "run_id": RUN_ID,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_candidate_zip_sha256": SOURCE_SHA,
                "official_accuracy_verified": False, "official_performance_verified": False,
                "official_acceptance": False, "submitted": False, "submission_ready": False,
                "files": [{"path": name, "bytes": len(raw), "sha256": sha256(raw)}
                          for name, raw in sorted(payload.items())]}
    payload["delivery-manifest.json"] = document(manifest)
    args.destination.mkdir(parents=True, exist_ok=False)
    for name, raw in payload.items():
        path = args.destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
    integrity = verify(args.destination)
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(payload.items()):
            archive.writestr(name, raw)
    zipped, identity = checked_zip(args.output)
    if zipped != payload:
        raise ValueError("Final ZIP content differs from reviewed delivery")
    print(json.dumps({"target": str(args.destination), "archive": identity,
                      "integrity": integrity, "frozen004_modified": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
