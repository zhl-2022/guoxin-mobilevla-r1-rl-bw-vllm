#!/usr/bin/env python3
"""Read-only native API + actual adapter processor check before model/GPU load.

Requires the isolated tool directory on PYTHONPATH and the small original RGB
checkpoint files. It does not instantiate vLLM/LLM/SigLIP model weights, install
packages or modify native code. A failure preserves traceback and source hashes.
"""
from __future__ import annotations

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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Native probe outputs must be new")
    report = {"schema": "mobilevla-native-api-processor-probe006-v1",
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "source": str(args.source), "model_weights_loaded": False,
              "official_accuracy_verified": False, "official_performance_verified": False,
              "status": "failed", "native_source_files": {}}
    try:
        for group in ("vllm.platform_plugins", "vllm.general_plugins"):
            report[group] = [{"name": x.name, "value": x.value}
                             for x in importlib.metadata.entry_points(group=group)]
        from mobilevla_bw006 import register
        register()
        from mobilevla_bw006 import backend
        from mobilevla_bw006.contract import IMAGE_TOKEN_ID, tokenize_source_prompt
        from mobilevla_bw006.reference_rgb import source_prompt
        from transformers import AutoTokenizer, LlavaConfig, SiglipImageProcessor
        import torch
        from PIL import Image
        for module in (
            backend,
            __import__("vllm.model_executor.models.llava", fromlist=["_"]),
            __import__("vllm.model_executor.models.siglip", fromlist=["_"]),
            __import__("vllm.model_executor.model_loader.default_loader", fromlist=["_"]),
            __import__("vllm.multimodal.processing.processor", fromlist=["_"]),
        ):
            path = Path(inspect.getsourcefile(module))
            report["native_source_files"][module.__name__] = {
                "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        config = LlavaConfig(
            text_config=json.loads((args.source / "llm/config.json").read_text()),
            vision_config=json.loads((args.source / "vision_tower/config.json").read_text()),
            image_token_index=IMAGE_TOKEN_ID,
            vision_feature_layer=-2,
            vision_feature_select_strategy="full",
            mobilevla_source_checkpoint=str(args.source.resolve()),
        )
        report["component_config_types"] = {
            "root": type(config).__name__, "text": type(config.text_config).__name__,
            "vision": type(config.vision_config).__name__,
        }
        if type(config.vision_config).__name__ != "SiglipVisionConfig":
            raise TypeError("Native Transformers does not create the required SigLIP config")
        tokenizer = AutoTokenizer.from_pretrained(str(args.source / "llm"), local_files_only=True)
        info = object.__new__(backend.MobileVLAProcessingInfo)
        info.ctx = SimpleNamespace(get_hf_config=lambda *a: config,
                                   get_tokenizer=lambda: tokenizer)
        adapter = object.__new__(backend.MobileVLAMultiModalProcessor)
        adapter.info = info
        images = [Image.new("RGB", (53, 29), (193, 7, 41)),
                  Image.new("L", (17, 31), 113)]
        prompt = source_prompt("Describe the frames.", len(images))
        actual = adapter._call_hf_processor(prompt, {"images": images}, {}, {})
        expected = SiglipImageProcessor.from_pretrained(
            str(args.source / "vision_tower"), local_files_only=True).preprocess(
                [image.convert("RGB").resize((384, 384)) for image in images], return_tensors="pt")["pixel_values"]
        report["actual_processor"] = {
            "pixel_shape": list(actual["pixel_values"].shape),
            "pixels_equal_reference": bool(torch.equal(actual["pixel_values"], expected)),
            "input_ids_equal_source": actual["input_ids"][0].tolist() == tokenize_source_prompt(prompt, tokenizer),
            "image_sentinel_count": actual["input_ids"][0].tolist().count(IMAGE_TOKEN_ID),
            "tokens_per_frame": info.get_num_image_tokens(image_width=53, image_height=29),
            "hf_processor_applies_updates": adapter._hf_processor_applies_updates(),
        }
        if not (report["actual_processor"]["pixels_equal_reference"]
                and report["actual_processor"]["input_ids_equal_source"]
                and report["actual_processor"]["image_sentinel_count"] == 2
                and report["actual_processor"]["tokens_per_frame"] == 196
                and report["actual_processor"]["hf_processor_applies_updates"] is False):
            raise ValueError("Actual native adapter processor contract differs")
        source = backend.DefaultModelLoader.Source("/unused/vision_tower", revision=None,
                                                   prefix="vision_tower.", fall_back_to_pt=False)
        report["secondary_source_native_constructor"] = {
            "type": type(source).__name__, "prefix": source.prefix,
        }
        report["status"] = "passed_api_and_cpu_processor_only"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    finally:
        report["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed_api_and_cpu_processor_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
