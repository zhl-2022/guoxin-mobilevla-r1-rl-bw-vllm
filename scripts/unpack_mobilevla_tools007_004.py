#!/usr/bin/env python3
"""Verify and install the exact small comparison bundle into a new directory."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

EXPECTED_ZIP = "f248a692df9ae69af29dae264e4ef6e659269593b25057b26eb17e18e2ad2336"
EXPECTED_MANIFEST = "08b0c2d0ff90b3651db278b6b531593487b2290905accae39a8358786b299043"
DESTINATION = Path("/root/private_data/guoxin-mobilevla-tools-007-004")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    source = Path(sys.argv[1] if len(sys.argv) == 2 else "/root/session007-mobilevla-tools-004.zip")
    if sha(source.read_bytes()) != EXPECTED_ZIP:
        raise ValueError("Expected exact frozen comparison ZIP identity")
    if DESTINATION.exists() or DESTINATION.is_symlink():
        raise FileExistsError("Comparison destination must be new; never overwrite")
    if not DESTINATION.parent.is_dir() or any(path.is_symlink() for path in DESTINATION.parents):
        raise ValueError("Expected existing regular destination ancestors")
    with zipfile.ZipFile(source) as archive:
        manifest_raw = archive.read("TOOLS-MANIFEST-007-004.json")
        if sha(manifest_raw) != EXPECTED_MANIFEST:
            raise ValueError("Expected exact frozen comparison manifest")
        manifest = json.loads(manifest_raw)
        files = manifest["files"]
        if len(archive.infolist()) != 105 or len(files) != 104 or archive.testzip() is not None:
            raise ValueError("Comparison ZIP member count or CRC differs")
        if set(archive.namelist()) != set(files) | {"TOOLS-MANIFEST-007-004.json"}:
            raise ValueError("Comparison ZIP member names differ")
        if sum(row["bytes"] for row in files.values()) > 30 * 1024 * 1024:
            raise ValueError("Comparison payload exceeds thirty MiB")
        payload = {}
        for name, expected in files.items():
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts or "\\" in name or ":" in name:
                raise ValueError("Unsafe comparison member")
            info = archive.getinfo(name)
            raw = archive.read(name)
            if ((info.external_attr >> 16) & 0o170000 != 0o100000 or
                    info.file_size != expected["bytes"] or len(raw) != expected["bytes"] or sha(raw) != expected["sha256"]):
                raise ValueError("Comparison payload identity/type differs")
            payload[name] = raw
    DESTINATION.mkdir()
    for name, raw in payload.items():
        path = DESTINATION.joinpath(*PurePosixPath(name).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
        path.chmod(0o755 if name.endswith(".sh") else 0o644)
        if sha(path.read_bytes()) != files[name]["sha256"]:
            raise ValueError("Installed comparison readback differs")
    record = {"schema": "mobilevla-native-alignment-tools-unpack007-eos004-v1", "zip_sha256": EXPECTED_ZIP,
              "manifest_sha256": EXPECTED_MANIFEST, "payload_files": len(payload),
              "all_installed_files_readback_verified": True, "GPU_used": False,
              "destination": str(DESTINATION)}
    with (DESTINATION / "unpack-ready-007-004.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
