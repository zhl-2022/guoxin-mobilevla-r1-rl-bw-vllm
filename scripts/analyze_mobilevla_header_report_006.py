#!/usr/bin/env python3
"""Offline structural review of a real bounded MobileVLA header report.

No model/GPU imports; validates only actual header names/shapes/dtypes. It cannot
turn a partial header into full-file integrity, model loading or accuracy proof.
"""
import argparse
import json
from pathlib import Path


def analyze(report):
    if report.get("schema") != "mobilevla-bounded-header-inspection006-v1":
        raise ValueError("Expected actual bounded header inspection report")
    files = report.get("files", [])
    by_name = {x["path"]: x for x in files}
    if len(files) != 6 or len(by_name) != 6:
        raise ValueError("Expected six selected RL safetensors entries")
    problems = []
    unavailable = [x["path"] for x in files if x["status"] != "header_parsed"]
    if unavailable:
        return {"status": "headers_incomplete", "unavailable": unavailable, "problems": [],
                "GPU_verified": False, "full_file_integrity_verified": False}
    llm = {}
    for path, item in by_name.items():
        if "/llm/" in path:
            overlap = set(llm) & set(item["tensors"])
            if overlap:
                problems.append("Duplicate LLM keys: " + str(sorted(overlap)))
            llm.update(item["tensors"])
    expected = {"model.embed_tokens.weight": [128259, 4096], "lm_head.weight": [128259, 4096],
                "model.norm.weight": [4096]}
    for i in range(32):
        prefix = f"model.layers.{i}."
        expected.update({prefix + name: shape for name, shape in {
            "input_layernorm.weight": [4096], "post_attention_layernorm.weight": [4096],
            "self_attn.q_proj.weight": [4096, 4096],
            "self_attn.k_proj.weight": [1024, 4096],
            "self_attn.v_proj.weight": [1024, 4096],
            "self_attn.o_proj.weight": [4096, 4096],
            "mlp.gate_proj.weight": [14336, 4096],
            "mlp.up_proj.weight": [14336, 4096],
            "mlp.down_proj.weight": [4096, 14336],
        }.items()})
    if set(llm) != set(expected):
        problems.append("LLM exact291 key set differs")
    for name, shape in expected.items():
        if llm.get(name, {}).get("shape") != shape:
            problems.append("LLM tensor shape differs: " + name)
    projector = by_name["weight/rl/mm_projector/model.safetensors"]["tensors"]
    expected_projector = {"layers.1.weight": [4608], "layers.1.bias": [4608],
                          "layers.2.weight": [4096, 4608], "layers.2.bias": [4096],
                          "layers.4.weight": [4096, 4096], "layers.4.bias": [4096]}
    if {k: v["shape"] for k, v in projector.items()} != expected_projector:
        problems.append("Projector six tensor names/shapes differ")
    vision = by_name["weight/rl/vision_tower/model.safetensors"]["tensors"]
    for name, shape in {
        "vision_model.embeddings.patch_embedding.weight": [1152, 3, 14, 14],
        "vision_model.embeddings.patch_embedding.bias": [1152],
        "vision_model.embeddings.position_embedding.weight": [729, 1152],
    }.items():
        if vision.get(name, {}).get("shape") != shape:
            problems.append("SigLIP embedding tensor differs: " + name)
    for layer in range(27):
        prefix = f"vision_model.encoder.layers.{layer}."
        for suffix, shape in {"layer_norm1.weight": [1152], "layer_norm2.weight": [1152],
                              "self_attn.q_proj.weight": [1152, 1152],
                              "self_attn.k_proj.weight": [1152, 1152],
                              "self_attn.v_proj.weight": [1152, 1152],
                              "self_attn.out_proj.weight": [1152, 1152],
                              "mlp.fc1.weight": [4304, 1152],
                              "mlp.fc2.weight": [1152, 4304]}.items():
            if vision.get(prefix + suffix, {}).get("shape") != shape:
                problems.append("SigLIP layer tensor differs: " + prefix + suffix)
    dtypes = sorted({v["dtype"] for item in files for v in item["tensors"].values()})
    if any(x not in {"BF16", "F32"} for x in dtypes):
        problems.append("Unexpected published checkpoint dtype: " + str(dtypes))
    return {"schema": "mobilevla-real-header-structural-analysis006-v1",
            "status": "headers_match_expected_structure" if not problems else "header_mismatch",
            "problems": problems, "llm_tensor_count": len(llm), "projector_tensor_count": len(projector),
            "vision_tensor_count": len(vision), "dtypes": dtypes,
            "header_sources": {x["path"]: {"source": x.get("source"), "header_sha256": x.get("header_sha256")}
                               for x in files},
            "GPU_verified": False, "full_file_integrity_verified": False,
            "official_accuracy_verified": False, "submission_ready": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(json.loads(args.report.read_text()))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "headers_match_expected_structure" else 2


if __name__ == "__main__":
    raise SystemExit(main())
