"""Bounded native vLLM service using the existing owned-process supervisor."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys

from . import ARCHITECTURE
from .contract import DERIVED_SMALL_FILES, IMAGE_TOKEN_ID, REVISION, sha256_file

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "scripts"))
import recon2reason_runtime as supervisor

TASK = "mobilevla-r1-rl-bw-vllm"


def configuration(task):
    model = Path(os.environ.get("MOBILE_MODEL_PATH", "/root/private_data/guoxin-candidates/models/mobilevla-r1-derived007-eos004")).expanduser()
    if any(path.is_symlink() for path in (model, *model.parents)):
        raise ValueError("Derived model directory and ancestors must not be symlinks")
    model = model.resolve()
    port = int(os.environ.get("MOBILE_PORT", "30139"))
    memory = float(os.environ.get("MOBILE_MEM_FRAC", "0.80"))
    load = int(os.environ.get("MOBILE_LOAD_TIMEOUT", "900"))
    runtime = int(os.environ.get("MOBILE_MAX_RUNTIME_SECONDS", "1800"))
    if not 1024 <= port <= 65535 or not 0 < memory < 1 or not 60 <= load <= runtime <= 6900:
        raise ValueError("Invalid bounded MobileVLA pilot configuration")
    served = "mobilevla-r1-rgb-rl006"
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(model),
               "--tokenizer", str(model), "--served-model-name", served, "--host", "127.0.0.1", "--port", str(port),
               "--dtype", "bfloat16", "--tensor-parallel-size", "1", "--max-model-len", "4096",
               "--gpu-memory-utilization", str(memory), "--max-num-seqs", "1", "--max-num-batched-tokens", "4096",
               "--enforce-eager", "--limit-mm-per-prompt", '{"image":8,"video":0}',
               "--chat-template", str(model / "mobilevla-chat-template.jinja"),
               "--chat-template-content-format", "string", "--generation-config", "vllm"]
    return {"task": str(Path(task).resolve()), "framework": "vllm", "precision": "bf16",
            "model_id": "AIGeeksGroup/MobileVLA-R1", "model_revision": REVISION,
            "contest_alias": "BAAI/MobileVLA-R1", "contest_alias_equivalence_verified": False,
            "model_path": str(model), "served_name": served, "host": "127.0.0.1", "port": port,
            "tp": 1, "max_len": 4096, "mem_frac": memory, "max_num_seqs": 1,
            "load_timeout": load, "max_runtime": runtime, "stop_grace": 30,
            "command": command, "url": f"http://127.0.0.1:{port}",
            "plugin": "mobilevla_bw006", "architecture": ARCHITECTURE,
            "scope": "Published RGB RL checkpoint; no depth/point modules or fabricated evaluation",
            "GPU_verified": False, "official_accuracy_verified": False,
            "official_performance_verified": False, "standard_submission_ready": False}


def checkpoint(model, hash_weights=False):
    model = Path(model)
    if any(path.is_symlink() for path in (model, *model.parents)):
        raise ValueError("Derived model directory and ancestors must not be symlinks")
    model = model.resolve(strict=True)
    record = json.loads((model / "prepare-receipt.json").read_text())
    config = json.loads((model / "config.json").read_text())
    if config.get("architectures") != [ARCHITECTURE] or config.get("image_token_index") != IMAGE_TOKEN_ID:
        raise ValueError("Derived model must select actual dedicated MobileVLA adapter")
    if record.get("schema") != "mobilevla-derived-hf-view-007-eos004-v1" or record.get("source_unchanged") is not True:
        raise ValueError("Derived checkpoint does not have a successful source seal")
    if record.get("adapter_revision") != "007-eos004" or config.get("mobilevla_adapter_revision") != "007-eos004":
        raise ValueError("Derived model does not belong to the 007-eos004 adapter")
    generation = json.loads((model / "generation_config.json").read_text())
    if (config.get("eos_token_id") != 128009 or config.get("text_config", {}).get("eos_token_id") != 128009
            or generation.get("eos_token_id") != 128009
            or record.get("publisher_eos_mapping", {}).get("derived_root_text_and_generation_eos_token_id") != 128009):
        raise ValueError("Derived primary/text/generation EOS must match publisher explicit128009")
    adapter_sources = {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    if record.get("adapter_source_files") != adapter_sources:
        raise ValueError("Derived receipt adapter source identities differ")
    if record.get("derived") != str(model):
        raise ValueError("Derived model path differs from its preparation identity")
    derived_seal = record.get("derived_small_files", {})
    if set(derived_seal) != DERIVED_SMALL_FILES:
        raise ValueError("Derived receipt must seal all eight generated small files")
    derived_rows = []
    for name, expected in sorted(derived_seal.items()):
        if (not isinstance(expected, dict) or set(expected) != {"bytes", "sha256"}
                or type(expected["bytes"]) is not int or expected["bytes"] <= 0
                or not isinstance(expected["sha256"], str)
                or not re.fullmatch("[0-9a-f]{64}", expected["sha256"])):
            raise ValueError(f"Invalid generated-file seal: {name}")
        path = model / name
        regular = path.is_file() and not path.is_symlink()
        size_ok = regular and path.stat().st_size == expected["bytes"]
        actual = sha256_file(path) if size_ok else None
        derived_rows.append({"path": name, "size_ok": size_ok,
                             "actual_sha256": actual, "identity_ok": actual == expected["sha256"]})
    if not all(x["identity_ok"] for x in derived_rows):
        return {"model": str(model), "weights_hashed": hash_weights,
                "derived_small_files": derived_rows, "files": [], "complete": False,
                "reason": "derived small file no longer matches prepare receipt", "GPU_verified": False}
    source = Path(record["source"])
    if any(path.is_symlink() for path in (source, *source.parents)):
        raise ValueError("Source checkpoint directory and ancestors must not be symlinks")
    if str(source) != config.get("mobilevla_source_checkpoint"):
        raise ValueError("Native secondary source differs from verified original checkpoint")
    rows = []
    for name, expected in record["source_files"].items():
        path = source / name
        size_ok = path.is_file() and not path.is_symlink() and path.stat().st_size == expected["bytes"]
        row = {"path": name, "size_ok": size_ok, "identity_ok": False}
        if size_ok and (hash_weights or not name.endswith(".safetensors")):
            row["actual_sha256"] = sha256_file(path)
            row["identity_ok"] = row["actual_sha256"] == expected["sha256"]
        if name.startswith("llm/") and name.endswith(".safetensors") and size_ok:
            derived = model / Path(name).name
            row["same_hardlinked_source"] = derived.is_file() and os.path.samefile(derived, path)
        rows.append(row)
    return {"source": str(source), "model": str(model), "weights_hashed": hash_weights,
            "files": rows, "derived_small_files": derived_rows,
            "complete": len(rows) == 16 and all(
                r["identity_ok"] and r.get("same_hardlinked_source", True) for r in rows),
            "full_lfs_parts_sha_verified": False, "GPU_verified": False}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("launch", "stop", "env", "preflight", "verify", "bench"))
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--hash-weights", action="store_true")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args(argv)
    task = args.task_dir.resolve()
    data = json.loads((task / "task.json").read_text())
    if task.name != TASK or data.get("model_revision") != REVISION:
        raise ValueError("Unexpected MobileVLA task identity")
    supervisor.checked_run_id(args.run_id)
    if args.action == "stop":
        if args.dry_run:
            print(json.dumps({"status": "plan_only", "run_id": args.run_id, "process_signalled": False}))
            return 0
        return supervisor.stop(task, args.run_id)
    cfg = configuration(task)
    if args.action == "preflight":
        result = checkpoint(cfg["model_path"], args.hash_weights)
        print(json.dumps(result, indent=2))
        return 0 if result["complete"] else 2
    if args.action == "launch" and not args.dry_run:
        if sys.platform != "linux":
            return supervisor.fail("Actual serving requires the SCNet Linux image")
        check = checkpoint(cfg["model_path"], hash_weights=True)
        if not check["complete"]:
            print(json.dumps(check, indent=2))
            return 2
        plugin_path = str(Path(__file__).resolve().parents[1])
        os.environ["PYTHONPATH"] = plugin_path + os.pathsep + os.environ.get("PYTHONPATH", "")
        # VLLM_PLUGINS filters platform plugins too. Restricting it to this
        # general plugin would disable the native DAS/BW platform plugin.
        if os.environ.get("VLLM_PLUGINS"):
            allowed = os.environ["VLLM_PLUGINS"].split(",")
            os.environ["VLLM_PLUGINS"] = ",".join(dict.fromkeys(allowed + ["mobilevla_bw006"]))
    original = supervisor.configuration
    supervisor.configuration = lambda _task: cfg
    try:
        if args.action == "launch":
            return supervisor.launch(task, args.dry_run, args.run_id)
        if args.action == "env" and args.dry_run:
            packages = {}
            for name in ("torch", "vllm", "vllm-hcu", "transformers", "Pillow"):
                try:
                    packages[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    packages[name] = None
            print(json.dumps({"configuration": cfg, "metadata_only": True, "packages": packages}, indent=2))
            return 0
        if args.action == "env":
            return supervisor.environment(task)
        if args.action in ("verify", "bench"):
            # Developer pilot profile is4096, cannot claim32K official matrix.
            if args.action == "bench" and not args.dry_run:
                print(json.dumps({"status": "pilot_profile_not_full_T02_load", "executed": False,
                                  "max_model_len": 4096, "official_performance_verified": False}))
                return 2
            return supervisor.official(task, args.action, args.dry_run, args.selfcheck)
    finally:
        supervisor.configuration = original
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
