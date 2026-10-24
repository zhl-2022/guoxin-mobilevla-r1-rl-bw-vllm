#!/usr/bin/env python3
"""Verify all sealed source/evidence bytes in an extracted tree or Git clone."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess

MANIFEST = "STANDARD-MANIFEST-007-004.json"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--check-git-index", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("Repository root must have regular ancestors")
    raw = (root / MANIFEST).read_bytes()
    if sha(raw) != args.manifest_sha256:
        raise ValueError("Manifest differs from independent package receipt")
    manifest = json.loads(raw)
    if manifest["adapter_revision"] != "007-eos004" or manifest["development_evidence_sealed"] is not True:
        raise ValueError("Expected the sealed004 development deliverable")
    if manifest.get("native_task_GPU_replayed") is not True or manifest.get("standard_wrapper_GPU_replayed") is not False:
        raise ValueError("004 native task/wrapper GPU evidence scope differs")
    expected = manifest["files"]
    for name, row in expected.items():
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts or "\\" in name or ":" in name:
            raise ValueError("Unsafe source inventory path")
        path = root.joinpath(*pure.parts)
        if path.is_symlink() or not path.is_file():
            raise ValueError("Missing or linked inventory member: " + name)
        content = path.read_bytes()
        if len(content) != row["bytes"] or sha(content) != row["sha256"]:
            raise ValueError("Inventory member bytes differ: " + name)
    actual = set()
    def onerror(error):
        raise error
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=onerror):
        for name in list(dirs):
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError("Linked repository directory")
            if directory == str(root) and name == ".git":
                dirs.remove(name)
        for name in files:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError("Linked repository file")
            actual.add(path.relative_to(root).as_posix())
    if actual != set(expected) | {MANIFEST}:
        raise ValueError("Repository membership differs: " + json.dumps({"extra": sorted(actual - set(expected) - {MANIFEST}), "missing": sorted(set(expected) - actual)}))
    result = {"passed": True, "manifest_sha256": sha(raw), "payload_files": len(expected),
              "all_payload_SHA256_checked": True, "exact_file_membership_checked": True,
              "Git_index_checked": False, "GPU_used": False, "submitted": False,
              "official_accuracy_verified": False, "official_performance_verified": False}
    if args.check_git_index:
        command = lambda *parts: subprocess.check_output(["git", "-C", str(root), *parts], timeout=30)
        tracked = {name.decode("utf-8") for name in command("ls-files", "-z").split(b"\0") if name}
        if tracked != actual:
            raise ValueError("Git tracked file set differs from sealed source inventory")
        result.update(Git_index_checked=True, commit=command("rev-parse", "HEAD").decode().strip(),
                      branch=command("rev-parse", "--abbrev-ref", "HEAD").decode().strip())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
