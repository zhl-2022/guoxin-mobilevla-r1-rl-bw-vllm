"""Verify the exact reviewed source tree, preserving read errors and old evidence."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat

ROOT = Path(__file__).resolve().parent
MANIFEST = "DELIVERY-MANIFEST-008.json"


def reject(error):
    raise error


def main():
    rows = json.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))["files"]
    expected = {r["path"]: r for r in rows}
    if len(expected) != len(rows):
        raise ValueError("Duplicate manifest entries")
    actual = set()
    for directory, dirs, files in os.walk(ROOT, topdown=True, onerror=reject, followlinks=False):
        here = Path(directory)
        for name in list(dirs):
            p = here / name
            if here == ROOT and name == ".git" and p.is_dir() and not p.is_symlink():
                dirs.remove(name)
            elif p.is_symlink():
                raise ValueError("Unexpected directory link: " + str(p))
        for name in files:
            p = here / name
            if not stat.S_ISREG(p.lstat().st_mode):
                raise ValueError("Nonregular file: " + str(p))
            actual.add(p.relative_to(ROOT).as_posix())
    if actual != set(expected) | {MANIFEST}:
        raise ValueError("Exact file set mismatch")
    for name, row in expected.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or ":" in name or "\\" in name:
            raise ValueError("Invalid manifest path")
        raw = (ROOT / name).read_bytes()
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("File identity mismatch: " + name)
    print(json.dumps({"integrity_passed": True, "files": len(actual), "GPU_executed": False,
                      "official_accuracy_verified": False, "submission_ready": False}))


if __name__ == "__main__":
    main()
