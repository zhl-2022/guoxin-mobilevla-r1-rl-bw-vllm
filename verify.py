#!/usr/bin/env python3
"""Delegate unchanged official-material gate; no fabricated result."""
import os, sys
from pathlib import Path
entry=Path(__file__).resolve().parent / "tasks/mobilevla-r1-rl-bw-vllm-standard007/verify.py"
os.execv(sys.executable,[sys.executable,str(entry),*sys.argv[1:]])
