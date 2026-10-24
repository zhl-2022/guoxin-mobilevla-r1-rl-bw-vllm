#!/usr/bin/env python3
"""Delegate the draft standard entry point to the frozen official-data gate."""
from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    task = Path(__file__).resolve().parent.parent / "mobilevla-r1-rl-bw-vllm"
    raise SystemExit(subprocess.call(["bash", str(task / "verify.sh"), *sys.argv[1:]]))
