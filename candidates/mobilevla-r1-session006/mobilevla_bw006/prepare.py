"""Create a small derived HF view; checkpoint bytes stay unchanged.

Run on the CPU before starting GPU service. Hard links for four LLM shards save
16GB of duplicated disk storage; native secondary sources load vision/projector.
Every source file is SHA256 sealed and safetensors headers are checked without
deserializing model objects or using pickle. Refuse an existing output directory.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import struct

from . import ARCHITECTURE
from .contract import (DERIVED_SMALL_FILES, IMAGE_TOKEN, IMAGE_TOKEN_ID, REVISION, SOURCE_COMMIT,
                       chat_template, sha256_file, validate_checkpoint_config)


def read_safetensors_header(path):
    with path.open("rb") as stream:
        first = stream.read(8)
        if len(first) != 8:
            raise ValueError(f"Missing safetensors length: {path}")
        size = struct.unpack("<Q", first)[0]
        if not 2 <= size <= 16 * 1024 * 1024:
            raise ValueError("Unbounded safetensors JSON header")
        data = stream.read(size)
        if len(data) != size:
            raise ValueError("Truncated safetensors JSON header")
    header = json.loads(data)
    data_bytes = path.stat().st_size - size - 8
    ranges = []
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(tensor, dict) or tensor.get("dtype") not in ("BF16", "F32"):
            raise ValueError(f"Unexpected dtype at {path}:{name}")
        shape = tensor.get("shape")
        offsets = tensor.get("data_offsets")
        if not isinstance(shape, list) or not all(type(v) is int and v > 0 for v in shape):
            raise ValueError("Invalid tensor shape")
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(type(x) is int for x in offsets):
            raise ValueError("Invalid tensor offsets")
        count = 1
        for dimension in shape:
            count *= dimension
        byte_width = 2 if tensor["dtype"] == "BF16" else 4
        if not 0 <= offsets[0] < offsets[1] <= data_bytes or offsets[1] - offsets[0] != count * byte_width:
            raise ValueError("Tensor bounds/shape/dtype do not match")
        ranges.append(tuple(offsets))
    ranges.sort()
    if not ranges or ranges[0][0] != 0 or ranges[-1][1] != data_bytes:
        raise ValueError("Unaccounted safetensors data")
    if any(a[1] != b[0] for a, b in zip(ranges, ranges[1:])):
        raise ValueError("Overlapping or missing safetensors payload")
    return header


def validate_headers(source):
    index = json.loads((source / "llm/model.safetensors.index.json").read_text())
    mapping = index["weight_map"]
    if len(mapping) != 291 or len(set(mapping.values())) != 4:
        raise ValueError("Expected exact four-shard Llama3 8B checkpoint")
    all_names = {}
    headers = {}
    for filename in sorted(set(mapping.values())):
        if Path(filename).name != filename or not filename.endswith(".safetensors"):
            raise ValueError("Invalid LLM shard name")
        header = read_safetensors_header(source / "llm" / filename)
        headers["llm/" + filename] = header
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if mapping.get(name) != filename or name in all_names:
                raise ValueError("LLM index/shard keys disagree")
            all_names[name] = tensor
    if set(mapping) != set(all_names):
        raise ValueError("LLM shard keys missing")
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        if all_names[name]["shape"] != [128259, 4096]:
            raise ValueError("Original expanded Llama embedding/head shape mismatch")
    projector = read_safetensors_header(source / "mm_projector/model.safetensors")
    expected = {"layers.1.weight": [4608], "layers.1.bias": [4608],
                "layers.2.weight": [4096, 4608], "layers.2.bias": [4096],
                "layers.4.weight": [4096, 4096], "layers.4.bias": [4096]}
    observed = {k: v["shape"] for k, v in projector.items() if k != "__metadata__"}
    if observed != expected:
        raise ValueError(f"Actual MobileVLA projector tensor names/shapes differ: {observed}")
    headers["mm_projector/model.safetensors"] = projector
    vision = read_safetensors_header(source / "vision_tower/model.safetensors")
    if vision["vision_model.embeddings.patch_embedding.weight"]["shape"] != [1152, 3, 14, 14]:
        raise ValueError("SigLIP patch embedding shape differs")
    if vision["vision_model.embeddings.position_embedding.weight"]["shape"] != [729, 1152]:
        raise ValueError("SigLIP position embedding shape differs")
    for layer in range(27):
        if vision[f"vision_model.encoder.layers.{layer}.layer_norm1.weight"]["shape"] != [1152]:
            raise ValueError("Missing actual SigLIP encoder layer")
    headers["vision_tower/model.safetensors"] = vision
    return headers


def seal_derived_small_files(output):
    """Seal only generated files; never include the receipt or hardlinked weights."""
    result = {}
    for name in sorted(DERIVED_SMALL_FILES):
        path = output / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Expected generated regular small file: {name}")
        result[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return result


def publisher_eos_configs(text_config, generation_config):
    """Map only derived stopping metadata to publisher's explicit generate()."""
    if text_config.get("eos_token_id") != 128009 or generation_config.get("eos_token_id") != [128001, 128009]:
        raise ValueError("Unexpected source EOS metadata; refuse an unreviewed mapping")
    return ({**text_config, "eos_token_id": 128009},
            {**generation_config, "eos_token_id": 128009})


def prepare(source: Path, output: Path, receipt_path: Path):
    source = source.resolve(strict=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Derived output must be a new directory")
    config = validate_checkpoint_config(source)
    headers = validate_headers(source)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("schema") != "mobilevla-selected-rgb-rl-receipt-006-v1"
            or receipt.get("repo") != "AIGeeksGroup/MobileVLA-R1"
            or receipt.get("revision") != REVISION
            or receipt.get("status") != "completed"
            or receipt.get("full_lfs_parts_sha_verified") is not False
            or receipt.get("verified_crc32_and_size") is not True):
        raise ValueError("Expected completed exact-revision partial-archive receipt")
    plan = json.loads((Path(__file__).resolve().parents[1] / "rl-entry-plan.json").read_text())
    if receipt.get("central_directory_sha256") != plan["central_directory_sha256"]:
        raise ValueError("Receipt central-directory identity differs from pinned plan")
    planned = {x["path"]: x for x in plan["entries"]}
    records = receipt.get("files", [])
    sealed = {x["path"]: x for x in records}
    if len(records) != 16 or len(sealed) != 16 or set(sealed) != set(planned):
        raise ValueError("Expected exact 16-file RGB RL extraction receipt")
    for name, record in sealed.items():
        for key, planned_key in (("size", "uncompressed_size"), ("crc32", "crc32"),
                                 ("compressed_size", "compressed_size"),
                                 ("compression_method", "compression_method")):
            if record.get(key) != planned[name][planned_key]:
                raise ValueError(f"Extraction receipt entry differs: {name}/{key}")
    names = ["config.json", "llm/config.json", "llm/generation_config.json",
             "llm/tokenizer.json", "llm/tokenizer_config.json", "llm/special_tokens_map.json",
             "llm/model.safetensors.index.json", "vision_tower/config.json",
             "vision_tower/preprocessor_config.json", "mm_projector/config.json"] + list(headers)
    before = {}
    for name in sorted(names):
        path = source / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Expected regular verified checkpoint file: {name}")
        before[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        extracted = sealed["weight/rl/" + name]
        if before[name] != {"bytes": extracted["size"], "sha256": extracted["sha256"]}:
            raise ValueError(f"Actual source file differs from verified extraction: {name}")
    output.mkdir(parents=True)
    for name in sorted(headers):
        if name.startswith("llm/"):
            # Same filesystem is deliberate: no silent 16GB copy fallback.
            os.link(source / name, output / Path(name).name)
    for name in ("model.safetensors.index.json", "generation_config.json", "special_tokens_map.json"):
        shutil.copyfile(source / "llm" / name, output / name)
    # vLLM still imports generation-config special tokens when ordinary
    # sampling defaults are disabled with --generation-config vllm. The
    # publisher passes tokenizer.eos_token_id=128009 explicitly to generate().
    text_config, generation = publisher_eos_configs(config["llm"], json.loads((source / "llm/generation_config.json").read_text(encoding="utf-8")))
    (output / "generation_config.json").write_text(json.dumps(generation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(source / "vision_tower/preprocessor_config.json", output / "preprocessor_config.json")
    tokenizer = json.loads((source / "llm/tokenizer.json").read_text(encoding="utf-8"))
    added = tokenizer["added_tokens"]
    if any(x["id"] >= IMAGE_TOKEN_ID or x["content"] == IMAGE_TOKEN for x in added):
        raise ValueError("Unexpected existing sentinel/vocabulary changes")
    special = {"id": IMAGE_TOKEN_ID, "content": IMAGE_TOKEN, "single_word": False,
               "lstrip": False, "rstrip": False, "normalized": False, "special": True}
    added.append(special)
    (output / "tokenizer.json").write_text(json.dumps(tokenizer, ensure_ascii=False) + "\n", encoding="utf-8")
    tokenizer_cfg = json.loads((source / "llm/tokenizer_config.json").read_text(encoding="utf-8"))
    tokenizer_cfg["added_tokens_decoder"][str(IMAGE_TOKEN_ID)] = {k: v for k, v in special.items() if k != "id"}
    (output / "tokenizer_config.json").write_text(json.dumps(tokenizer_cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Standard config object provides vLLM the correct component dimensions;
    # the registered custom class implements all Mobile-specific differences.
    derived = {
        "model_type": "llava", "architectures": [ARCHITECTURE],
        "text_config": text_config, "vision_config": config["vision"],
        "image_token_index": IMAGE_TOKEN_ID, "vision_feature_layer": -2,
        "vision_feature_select_strategy": "full", "projector_hidden_act": "gelu",
        "multimodal_projector_bias": True, "image_seq_length": 196,
        "torch_dtype": "bfloat16", "tie_word_embeddings": False,
        "bos_token_id": 128000, "eos_token_id": 128009, "pad_token_id": 128256,
        "mobilevla_source_checkpoint": str(source),
        "mobilevla_publisher_revision": REVISION,
        "mobilevla_publisher_code_commit": SOURCE_COMMIT,
        "mobilevla_source_config_sha256": before["config.json"]["sha256"],
        "mobilevla_adapter_scope": "RGB-only RL; native Llama/SigLIP + source 2x2 downsample projector",
        "mobilevla_adapter_revision": "007-eos004",
    }
    (output / "config.json").write_text(json.dumps(derived, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "mobilevla-chat-template.jinja").write_text(chat_template(), encoding="utf-8")
    # CPU-side postcondition: no source content has changed during preparation.
    for name, entry in before.items():
        path = source / name
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"Checkpoint source changed while preparing: {name}")
    derived_small_files = seal_derived_small_files(output)
    record = {
        "schema": "mobilevla-derived-hf-view-007-eos004-v1",
        "adapter_revision": "007-eos004",
        "publisher_eos_mapping": {"publisher_explicit_eos_token_id": 128009,
                                  "source_text_config_eos_token_id": config["llm"]["eos_token_id"],
                                  "source_generation_eos_token_id": json.loads((source / "llm/generation_config.json").read_text())["eos_token_id"],
                                  "derived_root_text_and_generation_eos_token_id": 128009,
                                  "source_files_modified": False},
        "adapter_source_files": {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": str(source), "derived": str(output.resolve()),
        "source_files": before, "source_unchanged": True,
        "derived_small_files": derived_small_files,
        "safetensors_headers": headers,
        "source_weights_were_not_rewritten": True,
        "extraction_receipt_sha256": sha256_file(receipt_path),
        "full_lfs_parts_sha_verified": False,
        "partial_archive_integrity_scope": plan["integrity"],
        "sentinel_adaptation": {"source": -200, "runtime": IMAGE_TOKEN_ID,
                                "no_embedding_rows_added": True},
        "gpu_loaded": False, "formal_ready": False, "submitted": False,
    }
    (output / "prepare-receipt.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.source, args.output, args.receipt)
    print(json.dumps({k: v for k, v in result.items() if k not in ("source_files", "safetensors_headers")}, indent=2))


if __name__ == "__main__":
    main()
