#!/usr/bin/env python3
"""Explicit-policy HF composition; execute only after the native service stops.

Uses unchanged original Llama/SigLIP weights and the frozen source-equivalent
projector. This helper is separate from the frozen 006-002 candidate.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time

from mobilevla_compare_policy006_004 import DEFAULT_QUESTION, generation_policy, validate_completion


def hf_generation_config(policy, config_class):
    # A fresh config prevents the checkpoint's sampling defaults leaking into
    # greedy reference generation. The publisher wrapper EOS is explicit.
    return config_class(bos_token_id=policy["bos_token_id"],
                        eos_token_id=policy["eos_token_ids"].copy(),
                        pad_token_id=policy["pad_token_id"],
                        do_sample=False, num_beams=1, max_new_tokens=policy["max_new_tokens"],
                        min_new_tokens=0, repetition_penalty=1.0, use_cache=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or not 1 <= len(args.image) <= 8:
        raise ValueError("Require new output and 1-8 PNG frames")
    from mobilevla_bw006.contract import IMAGE_TOKEN_ID, PROJECTED_TOKENS, sha256_file, tokenize_source_prompt, validate_checkpoint_config
    from mobilevla_bw006.reference_rgb import source_prompt
    source = args.source.resolve(strict=True)
    validate_checkpoint_config(source)
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, GenerationConfig, LlamaForCausalLM, SiglipImageProcessor, SiglipVisionModel
    from mobilevla_bw006.projector import MobileVLAProjector
    tokenizer = AutoTokenizer.from_pretrained(str(source / "llm"), local_files_only=True, use_fast=False, legacy=False)
    policy = generation_policy(source, tokenizer, args.max_new_tokens)
    prompt = source_prompt(args.question, len(args.image))
    ids = tokenize_source_prompt(prompt, tokenizer, image_token_id=-200)
    expanded = []
    for token in ids:
        expanded.extend([IMAGE_TOKEN_ID] * PROJECTED_TOKENS if token == -200 else [token])
    images, image_records = [], []
    for path in args.image:
        if path.suffix.lower() != ".png" or not path.is_file() or path.stat().st_size > 20 * 1024 * 1024:
            raise ValueError("Bounded comparison accepts local PNG frames only")
        if not path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("PNG signature is required")
        with Image.open(path) as frame:
            images.append(frame.convert("RGB").resize((384, 384)))
        image_records.append({"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    record = {"schema": "mobilevla-hf-comparison006-v4", "status": "failed",
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "source": str(source), "images": image_records, "question": args.question,
              "source_prompt": prompt, "source_prompt_token_ids": ids,
              "expected_expanded_prompt_token_ids": expanded, "generation_policy": policy,
              "official_accuracy_verified": False, "official_performance_verified": False,
              "scope": "Independent standard-component HF composition; no official baseline score"}
    started = time.monotonic()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Reference requires the reserved BW device after native service cleanup")
        image_processor = SiglipImageProcessor.from_pretrained(str(source / "vision_tower"), local_files_only=True)
        pixels = image_processor.preprocess(images, return_tensors="pt")["pixel_values"].to("cuda", torch.bfloat16)
        vision = SiglipVisionModel.from_pretrained(str(source / "vision_tower"), local_files_only=True,
                                                  dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        projector = MobileVLAProjector().to(device="cuda", dtype=torch.bfloat16).eval()
        projector.load_state_dict(load_file(str(source / "mm_projector/model.safetensors"), device="cpu"), strict=True)
        llm = LlamaForCausalLM.from_pretrained(str(source / "llm"), local_files_only=True,
                                              dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        with torch.inference_mode():
            raw_vision = vision(pixels, output_hidden_states=True).hidden_states[-2]
            visual = projector(raw_vision)
            chunks, frame_index = [], 0
            for token in ids:
                if token == -200:
                    chunks.append(visual[frame_index])
                    frame_index += 1
                else:
                    chunks.append(llm.get_input_embeddings()(torch.tensor([token], device="cuda", dtype=torch.long)))
            if frame_index != len(images):
                raise ValueError("Source sentinel/frame ordering differs")
            embeds = torch.cat(chunks, dim=0).unsqueeze(0)
            if list(embeds.shape) != [1, len(expanded), 4096]:
                raise ValueError("Unexpected expanded input embedding shape")
            output = llm.generate(inputs_embeds=embeds,
                                  attention_mask=torch.ones(embeds.shape[:2], dtype=torch.long, device="cuda"),
                                  generation_config=hf_generation_config(policy, GenerationConfig),
                                  return_dict_in_generate=True, output_scores=True)
            count = len(output.scores)
            if not count or output.sequences.shape[-1] < count:
                raise ValueError("HF sequences/scores do not expose complete generated token IDs")
            tokens = output.sequences[0, -count:].detach().cpu().tolist()
            finish = "stop" if tokens[-1] in policy["eos_token_ids"] else "length"
            problems = validate_completion(tokens, finish, None, policy)
            def digest(tensor):
                return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            record.update(generated_token_ids=tokens, generated_tokens=count, finish_reason=finish,
                          generated_text=tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                          pixel_shape=list(pixels.shape), pixel_tensor_sha256=digest(pixels),
                          vision_shape=list(raw_vision.shape), vision_tensor_sha256=digest(raw_vision),
                          visual_embedding_shape=list(visual.shape), visual_embedding_sha256=digest(visual),
                          expanded_input_embedding_shape=list(embeds.shape), comparison_problems=problems,
                          torch_version=torch.__version__, device_name=torch.cuda.get_device_name(0))
            record["status"] = "completed_comparable" if not problems else "completed_not_comparable"
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
        print(json.dumps({"output": str(args.output), "status": record["status"],
                          "generated_token_ids": record.get("generated_token_ids"), "error": record.get("error")}, ensure_ascii=False))
    return 0 if record["status"] == "completed_comparable" else 2


if __name__ == "__main__":
    raise SystemExit(main())
