#!/usr/bin/env python3
"""Compare immutable native/HF v2 records without loading a model."""
import argparse
import json
from pathlib import Path
from mobilevla_compare_policy006_003 import compare_records, sha256


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Comparison output must be new")
    result = compare_records(json.loads(args.native.read_text(encoding="utf-8")),
                             json.loads(args.reference.read_text(encoding="utf-8")))
    result["input_records"] = {"native": {"path": str(args.native.resolve()), "sha256": sha256(args.native)},
                               "reference": {"path": str(args.reference.resolve()), "sha256": sha256(args.reference)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["comparable"] and result["raw_generated_token_ids_equal"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
