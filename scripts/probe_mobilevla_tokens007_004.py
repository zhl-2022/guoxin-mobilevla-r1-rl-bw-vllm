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


def probe_eos_configuration(source, derived, tokenizer):
    """Execute installed vLLM configuration methods without model weights."""
    from vllm.config import ModelConfig
    from vllm.renderers.base import BaseRenderer
    from vllm.sampling_params import SamplingParams
    from vllm.transformers_utils.config import get_config, get_hf_text_config
    from transformers import AutoConfig
    import vllm.v1.engine.input_processor as input_processor
    fields = SimpleNamespace(model=str(derived), hf_config_path=None,
                             trust_remote_code=False, revision=None,
                             config_format="auto", generation_config="vllm",
                             override_generation_config={})
    generation = ModelConfig.try_get_generation_config(fields)
    fields.try_get_generation_config = lambda: generation
    sampling_defaults = ModelConfig.get_diff_sampling_param(fields)
    primary_eos = BaseRenderer.get_eos_token_id(SimpleNamespace(tokenizer=tokenizer))
    params = SamplingParams(temperature=0, max_tokens=128, stop_token_ids=[128009], ignore_eos=False)
    params.update_from_generation_config(generation, primary_eos)
    source_generation = json.loads((source / "llm/generation_config.json").read_text())
    historical = SamplingParams(temperature=0, max_tokens=128, stop_token_ids=[128009], ignore_eos=False)
    historical.update_from_generation_config(source_generation, primary_eos)
    config = json.loads((derived / "config.json").read_text())
    hf_auto = AutoConfig.from_pretrained(str(derived), local_files_only=True)
    hf_native = get_config(str(derived), trust_remote_code=False)
    object_eos = {"HF_AutoConfig_root": getattr(hf_auto, "eos_token_id", None),
                  "HF_AutoConfig_get_text_config": getattr(hf_auto.get_text_config(), "eos_token_id", None),
                  "vllm_get_config_root": getattr(hf_native, "eos_token_id", None),
                  "vllm_get_hf_text_config": getattr(get_hf_text_config(hf_native), "eos_token_id", None)}
    values = [config["eos_token_id"], config["text_config"]["eos_token_id"], generation["eos_token_id"], primary_eos]
    if (values != [128009] * 4 or set(object_eos.values()) != {128009}
            or params._eos_token_id != 128009 or params._all_stop_token_ids != {128009}):
        raise ValueError("Installed native EOS path does not match publisher explicit128009")
    if 128001 not in historical._all_stop_token_ids:
        raise ValueError("Installed native source-generation regression mechanism differs")
    sources = {}
    for label, obj in (("model_config", ModelConfig), ("renderer", BaseRenderer),
                       ("input_processor", input_processor), ("sampling_params", SamplingParams)):
        path = Path(inspect.getsourcefile(obj))
        sources[label] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {"status": "passed_publisher_EOS_configuration", "native_source_files": sources,
            "generation_config_mode": "vllm", "ordinary_sampling_defaults": sampling_defaults,
            "actual_generation_config_special_tokens": generation,
            "renderer_primary_eos_token_id": primary_eos, "effective_eos_token_id": params._eos_token_id,
            "effective_stop_token_ids": sorted(params._all_stop_token_ids),
            "historical003_source_generation_effective_stop_token_ids": sorted(historical._all_stop_token_ids),
            "derived_root_and_text_eos": values[:2], "actual_config_object_eos": object_eos,
            "source_original_LLM_config_eos": json.loads((source / "llm/config.json").read_text())["eos_token_id"],
            "source_original_generation_config_eos": source_generation["eos_token_id"],
            "model_weights_loaded": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--derived", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Token probe output must be new")
    record = {"schema": "mobilevla-native-publisher-token-path007-eos004-v1", "status": "failed",
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
        record["EOS_configuration_probe"] = probe_eos_configuration(args.source, args.derived, derived)
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
