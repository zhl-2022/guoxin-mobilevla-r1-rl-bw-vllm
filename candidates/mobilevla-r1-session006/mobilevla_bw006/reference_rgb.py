"""Independent HF RGB forward/generation reference for the pinned components.

This is a development reference, not the official evaluator. It composes the
original Llama/SigLIP checkpoint plus the source-equivalent projector and inserts
visual embeddings at upstream -200 placeholders. No vLLM code runs in this
process. Run only after the owned vLLM service has stopped and GPU is released.
No dependency installation or publisher environment_setup.sh is performed.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time

from .contract import SYSTEM_TEXT, sha256_file, tokenize_source_prompt, validate_checkpoint_config


def source_prompt(question, frames):
    # Exact Conversation(llama_3) layout validated against pinned AST in CPU tests.
    observations = "RGB observations:\n" + "<image>\n" * frames + question
    return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + SYSTEM_TEXT
            + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + observations
            + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--question", default="Describe the visible scene, then state a suitable navigation action in <answer>...</answer>.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= len(args.image) <= 8 or not 1 <= args.max_new_tokens <= 128:
        raise ValueError("Reference is bounded to1-8 images and1-128 completion tokens")
    if args.output.exists():
        raise FileExistsError("Reference output is immutable per run")
    source = args.source.resolve(strict=True)
    validate_checkpoint_config(source)
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, LlamaForCausalLM, SiglipImageProcessor, SiglipVisionModel
    from .projector import MobileVLAProjector

    if not torch.cuda.is_available():
        raise RuntimeError("Reference requires the reserved BW device")
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(str(source / "llm"), local_files_only=True)
    image_processor = SiglipImageProcessor.from_pretrained(str(source / "vision_tower"), local_files_only=True)
    images = [Image.open(path).convert("RGB").resize((384, 384)) for path in args.image]
    pixels = image_processor.preprocess(images, return_tensors="pt")["pixel_values"].to("cuda", torch.bfloat16)
    vision = SiglipVisionModel.from_pretrained(str(source / "vision_tower"), local_files_only=True,
                                              dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    projector = MobileVLAProjector().to(device="cuda", dtype=torch.bfloat16).eval()
    projector.load_state_dict(load_file(str(source / "mm_projector/model.safetensors"), device="cpu"), strict=True)
    llm = LlamaForCausalLM.from_pretrained(str(source / "llm"), local_files_only=True,
                                          dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    prompt = source_prompt(args.question, len(images))
    ids = tokenize_source_prompt(prompt, tokenizer, image_token_id=-200)
    with torch.inference_mode():
        raw_vision = vision(pixels, output_hidden_states=True).hidden_states[-2]
        visual_embeds = projector(raw_vision)
        input_ids = torch.tensor(ids, device="cuda", dtype=torch.long)
        chunks = []
        frame = 0
        for token in input_ids:
            if token.item() == -200:
                chunks.append(visual_embeds[frame])
                frame += 1
            else:
                chunks.append(llm.get_input_embeddings()(token.view(1)))
        if frame != len(images):
            raise ValueError("Source sentinel/frame ordering differs")
        embeds = torch.cat(chunks, dim=0).unsqueeze(0)
        output = llm.generate(inputs_embeds=embeds,
                              attention_mask=torch.ones(embeds.shape[:2], dtype=torch.long, device="cuda"),
                              max_new_tokens=args.max_new_tokens, do_sample=False,
                              use_cache=True, eos_token_id=tokenizer.eos_token_id,
                              pad_token_id=tokenizer.pad_token_id,
                              return_dict_in_generate=True, output_scores=True)
        count = len(output.scores)
        tokens = output.sequences[0, -count:].detach().cpu().tolist() if count else []
        def digest(tensor):
            return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        record = {
            "schema": "mobilevla-independent-HF-RGB-reference-006-v1",
            "ended_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "source": str(source), "framework": "transformers-standard-components",
            "torch_version": torch.__version__, "device_name": torch.cuda.get_device_name(0),
            "images": [{"path": str(p.resolve()), "sha256": sha256_file(p)} for p in args.image],
            "prompt": prompt, "source_prompt_token_ids": ids,
            "pixel_shape": list(pixels.shape), "pixel_tensor_sha256": digest(pixels),
            "vision_shape": list(raw_vision.shape), "vision_tensor_sha256": digest(raw_vision),
            "visual_embedding_shape": list(visual_embeds.shape), "visual_embedding_sha256": digest(visual_embeds),
            "expanded_input_embedding_shape": list(embeds.shape),
            "max_new_tokens": args.max_new_tokens, "generated_tokens": len(tokens),
            "generated_token_ids": tokens,
            "generated_text": tokenizer.decode(tokens, skip_special_tokens=True),
            "elapsed_seconds": time.monotonic() - started,
            "official_evaluator": False, "official_accuracy_passed": False,
            "reference_scope": "New composition matching pinned publisher RGB pipeline; not publisher wrapper execution or official baseline",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"output": str(args.output), "generated_token_ids": tokens,
                      "generated_text": record["generated_text"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
