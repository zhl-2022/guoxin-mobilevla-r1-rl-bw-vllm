"""Verify a sealed AuK final007 bundle without importing model frameworks.

Copyright scope notice: no new repository-wide open-source license selected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def fail_scan(error):
    raise error


def files_in(base, *, ignore_git=False):
    """Fail on inaccessible entries and links rather than silently omitting them."""
    found = set()
    for parent, directories, files in os.walk(base, onerror=fail_scan, followlinks=False):
        for name in directories + files:
            path = Path(parent) / name
            if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
                raise ValueError("Links are not allowed in the delivery: " + str(path))
        if ignore_git and Path(parent) == base and ".git" in directories:
            directories.remove(".git")
        for name in files:
            path = Path(parent) / name
            if not path.is_file():
                raise ValueError("Nonregular delivery member: " + str(path))
            found.add(path.relative_to(base).as_posix())
    return found


def verify(base):
    base = Path(base).resolve()
    manifest = json.loads((base / "delivery-manifest.json").read_text(encoding="utf-8"))
    expected = set()
    for row in manifest["files"]:
        name = row["path"]
        rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts or "\\" in name or name in expected:
            raise ValueError("Unsafe or duplicate manifest member: " + name)
        expected.add(name)
        path = base.joinpath(*rel.parts)
        if path.is_symlink() or not path.resolve().is_relative_to(base):
            raise ValueError("Unsafe delivery path: " + name)
        raw = path.read_bytes()
        if len(raw) != row["bytes"] or sha256(raw) != row["sha256"]:
            raise ValueError("Delivery member identity differs: " + name)
    actual = files_in(base, ignore_git=True)
    if actual != expected | {"delivery-manifest.json"}:
        raise ValueError("Delivery membership differs: " + repr({
            "unexpected": sorted(actual - expected - {"delivery-manifest.json"}),
            "missing": sorted(expected - actual),
        }))
    if any(manifest[key] is not False for key in (
        "official_accuracy_verified", "official_performance_verified",
        "official_acceptance", "submitted", "submission_ready",
    )):
        raise ValueError("Development bundle must preserve formal status boundaries")
    return {"passed": True, "members_verified": len(expected),
            "run_id": manifest["run_id"], "scope": "Exact sealed-file integrity only; no model execution or official scoring"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(verify(args.bundle), ensure_ascii=False))


if __name__ == "__main__":
    main()
