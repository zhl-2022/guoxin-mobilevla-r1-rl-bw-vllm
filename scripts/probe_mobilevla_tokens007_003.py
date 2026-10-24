#!/usr/bin/env python3
"""Native CPU tokenizer/token-path probe; never instantiate model weights."""
import argparse
import datetime
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--derived", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Token probe output must be new")
    record = {"schema": "mobilevla-native-publisher-token-path007-bos003-v1", "status": "failed",
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "model_weights_loaded": False, "official_accuracy_verified": False}
    try:
        from mobilevla_bw006 import register
        register()
        from mobilevla_bw006 import backend
        from mobilevla_bw006.contract import tokenize_source_prompt, IMAGE_TOKEN_ID
        from mobilevla_bw006.reference_rgb import source_prompt
        from transformers import AutoTokenizer
        original = AutoTokenizer.from_pretrained(str(args.source / "llm"), local_files_only=True, use_fast=False, legacy=False)
        derived = AutoTokenizer.from_pretrained(str(args.derived), local_files_only=True, use_fast=False, legacy=False)
        adapter = object.__new__(backend.MobileVLAMultiModalProcessor)
        adapter.info = SimpleNamespace(get_tokenizer=lambda: derived)
        adapter._apply_hf_processor_mm_only = lambda **kwargs: {}
        record["package_versions"] = {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers", "vllm", "vllm-hcu")}
        record["tokenizer_classes"] = {"publisher": type(original).__name__, "derived": type(derived).__name__}
        record["source_files"] = {}
        for label, value in (("backend", backend), ("native_tokens_main", adapter._apply_hf_processor_main),
                             ("native_tokenizer_class", type(derived))):
            path = Path(inspect.getsourcefile(value))
            record["source_files"][label] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        observations = []
        for frames in (1, 8):
            prompt = source_prompt("Describe the visible scene, then state a suitable navigation action in <answer>...</answer>.", frames)
            publisher_ids = tokenize_source_prompt(prompt, original, -200)
            incoming = derived.encode(prompt, add_special_tokens=False)
            actual, _, _ = adapter._apply_hf_processor_main(incoming, None, {}, {}, enable_hf_prompt_update=True)
            expected = [IMAGE_TOKEN_ID if x == -200 else x for x in publisher_ids]
            if actual != expected or publisher_ids[:2] != [128000, 128000]:
                raise ValueError("Native tokens-only path differs from publisher tokenizer flow")
            observations.append({"frames": frames, "publisher_ids": publisher_ids,
                                 "renderer_input_ids": incoming, "native_source_equivalent_ids": actual,
                                 "expanded_input_length": len(actual) + 195 * frames,
                                 "matches_publisher": True})
        record.update(observations=observations, status="passed_tokenizer_only")
    except Exception as exc:
        record.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"output": str(args.output), "status": record["status"], "error": record.get("error")}))
    return 0 if record["status"] == "passed_tokenizer_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
