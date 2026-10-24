#!/usr/bin/env python3
"""Inspect bounded safetensors headers from owned compressed cache or finals.

Useful while a large selected-RL download is still running. This never rewrites
the transfer/cache/model, deserializes tensor payload, imports Torch, or uses GPU.
Partial-cache header output is provisional and cannot prove full CRC/LFS identity.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import struct
import zlib

MAX_HEADER = 16 * 1024 * 1024
MAX_COMPRESSED_READ = 4 * 1024 * 1024


def compressed_header(path):
    decoder = zlib.decompressobj(-15)
    decoded = bytearray()
    pending = b""
    compressed_read = 0
    target = 8
    with path.open("rb") as stream:
        while len(decoded) < target:
            if not pending:
                pending = stream.read(min(65536, MAX_COMPRESSED_READ - compressed_read))
                compressed_read += len(pending)
                if not pending:
                    raise EOFError("Partial cache has not reached a complete bounded header")
            decoded.extend(decoder.decompress(pending, target - len(decoded)))
            pending = decoder.unconsumed_tail
            if len(decoded) == 8 and target == 8:
                length = struct.unpack("<Q", decoded)[0]
                if not 2 <= length <= MAX_HEADER:
                    raise ValueError("Invalid/unbounded safetensors header length")
                target = 8 + length
            if decoder.eof and len(decoded) < target:
                raise ValueError("Compressed stream ended inside safetensors header")
    return bytes(decoded[8:]), compressed_read


def final_header(path):
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise EOFError("Incomplete safetensors header length")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= MAX_HEADER:
            raise ValueError("Invalid/unbounded safetensors header length")
        raw = stream.read(length)
        if len(raw) != length:
            raise EOFError("Incomplete safetensors header")
    return raw


def inspect(plan, destination):
    if (plan.get("repo"), plan.get("revision"), plan.get("central_directory_sha256")) != (
        "AIGeeksGroup/MobileVLA-R1", "0c757f3e9d1b95d8416b42e7089e1c04f3e6118a",
        "4abb56a8a5710e7a7b2ec6664146ae57bc02d67c7b67a9eefb1bf4ae20a73373"
    ):
        raise ValueError("Expected pinned selected-RL plan")
    reports = []
    for row in plan["entries"]:
        path = PurePosixPath(row["path"])
        if (not row["path"].startswith("weight/rl/") or path.is_absolute()
                or ".." in path.parts or "\\" in row["path"] or ":" in row["path"]):
            raise ValueError("Only safe selected-RL paths are accepted")
        if not row["path"].endswith(".safetensors"):
            continue
        item = {"path": row["path"], "status": "not_available", "full_file_CRC_verified_by_this_probe": False}
        final = destination / row["path"]
        cache = destination / ".selected-compressed-006" / (hashlib.sha256(row["path"].encode()).hexdigest() + ".deflate")
        identity_file = cache.with_suffix(".identity.json")
        try:
            if final.is_symlink() or cache.is_symlink() or identity_file.is_symlink():
                raise ValueError("Only regular model/cache files are accepted")
            if final.is_file():
                raw = final_header(final)
                item["source"] = "final_file_header_only"
            elif cache.is_file() and identity_file.is_file():
                expected = {**row, "repo": plan["repo"], "revision": plan["revision"],
                            "central_directory_sha256": plan["central_directory_sha256"]}
                if json.loads(identity_file.read_text()) != expected:
                    raise ValueError("Compressed cache owner identity differs")
                raw, count = compressed_header(cache)
                item.update(source="owned_partial_compressed_cache", compressed_bytes_read=count,
                            provisional_header_not_full_integrity=True)
            else:
                reports.append(item)
                continue
            header = json.loads(raw)
            item.update(status="header_parsed", header_sha256=hashlib.sha256(raw).hexdigest(),
                        header_bytes=len(raw), tensors={k: v for k, v in header.items() if k != "__metadata__"},
                        safetensors_metadata=header.get("__metadata__"))
        except Exception as exc:
            item.update(status="header_unavailable_or_invalid", error=f"{type(exc).__name__}: {exc}")
        reports.append(item)
    return {"schema": "mobilevla-bounded-header-inspection006-v1", "files": reports,
            "model_weights_loaded": False, "GPU_imported": False,
            "full_file_CRC_verified_by_this_probe": False, "full_lfs_parts_sha_verified": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Header inspection outputs must be new")
    result = inspect(json.loads(args.plan.read_text()), args.destination)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"output": str(args.output), "files": [
        {"path": x["path"], "status": x["status"], "tensor_count": len(x.get("tensors", {})),
         "error": x.get("error")} for x in result["files"]]}, indent=2))


if __name__ == "__main__":
    main()
