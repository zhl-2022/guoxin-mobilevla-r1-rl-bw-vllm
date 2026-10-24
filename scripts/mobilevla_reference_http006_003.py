#!/usr/bin/env python3
"""One bounded native HTTP request with explicit same-input comparison policy."""
from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from mobilevla_compare_policy006_003 import DEFAULT_QUESTION, generation_policy, native_parameters, validate_completion


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--base-url", default="http://127.0.0.1:30139")
    parser.add_argument("--model", default="mobilevla-r1-rgb-rl006")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or not 1 <= len(args.image) <= 8:
        raise ValueError("Require new output and 1-8 PNG frames")
    url = urlsplit(args.base_url)
    if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost"}
            or url.username or url.password or url.path not in ("", "/") or url.query or url.fragment):
        raise ValueError("Only the owned loopback service base URL can be called")
    from mobilevla_bw006.contract import IMAGE_TOKEN_ID, PROJECTED_TOKENS, sha256_file, tokenize_source_prompt
    from mobilevla_bw006.reference_rgb import source_prompt
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.source / "llm"), local_files_only=True, use_fast=False, legacy=False)
    policy = generation_policy(args.source, tokenizer, args.max_new_tokens)
    prompt = source_prompt(args.question, len(args.image))
    original_ids = tokenize_source_prompt(prompt, tokenizer, image_token_id=-200)
    expected_ids = []
    for token in original_ids:
        expected_ids.extend([IMAGE_TOKEN_ID] * PROJECTED_TOKENS if token == -200 else [token])
    content = [{"type": "text", "text": "RGB observations:\n" + "<image>\n" * len(args.image) + args.question}]
    images = []
    for path in args.image:
        if path.suffix.lower() != ".png" or not path.is_file() or path.stat().st_size > 20 * 1024 * 1024:
            raise ValueError("Bounded comparison accepts local PNG frames only")
        raw = path.read_bytes()
        if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("PNG signature is required")
        content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(raw).decode()}})
        images.append({"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": len(raw)})
    body = {"model": args.model, "messages": [{"role": "user", "content": content}], **native_parameters(policy)}
    request_bytes = json.dumps(body).encode()
    record = {"schema": "mobilevla-native-comparison006-v3", "status": "failed",
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "images": images, "question": args.question, "source_prompt": prompt,
              "source_prompt_token_ids": original_ids, "expected_expanded_prompt_token_ids": expected_ids,
              "generation_policy": policy, "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
              "request_parameters_without_images": {k: v for k, v in body.items() if k != "messages"},
              "official_accuracy_verified": False, "official_performance_verified": False}
    started = time.monotonic()
    try:
        request = urllib.request.Request(args.base_url.rstrip("/") + "/v1/chat/completions", data=request_bytes,
                                         headers={"Content-Type": "application/json"}, method="POST")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=240) as response:
            raw = response.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise ValueError("Response exceeds bounded comparison limit")
            result = json.loads(raw)
        record["response"] = result
        choices = result.get("choices", [])
        if len(choices) != 1:
            raise ValueError("Expected exactly one native response choice")
        choice = choices[0]
        actual_ids = result.get("prompt_token_ids")
        tokens = choice.get("token_ids")
        usage = result.get("usage")
        record.update(actual_prompt_token_ids=actual_ids, generated_token_ids=tokens,
                      generated_text=choice.get("message", {}).get("content"), usage=usage,
                      finish_reason=choice.get("finish_reason"), stop_reason=choice.get("stop_reason"),
                      input_token_contract_matches=actual_ids == expected_ids)
        problems = validate_completion(tokens, choice.get("finish_reason"), choice.get("stop_reason"), policy,
                                       usage if usage is not None else {})
        if actual_ids != expected_ids:
            problems.append("expanded_prompt_token_ids_differ_or_missing")
        if not isinstance(usage, dict) or usage.get("prompt_tokens") != len(expected_ids):
            problems.append("usage_prompt_tokens_differs_from_expanded_input_length")
        record["comparison_problems"] = problems
        record["status"] = "completed_comparable" if not problems else "completed_not_comparable"
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, urllib.error.HTTPError):
            raw = exc.read(65536).decode("utf-8", "replace")
            record["error_body"] = re.sub(r'data:[^;\s\"]+;base64,[A-Za-z0-9+/=]+', "<image-data-redacted>", raw)
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
        print(json.dumps({"output": str(args.output), "status": record["status"],
                          "generated_token_ids": record.get("generated_token_ids"),
                          "comparison_problems": record.get("comparison_problems"), "error": record.get("error")}, ensure_ascii=False))
    return 0 if record["status"] == "completed_comparable" else 2


if __name__ == "__main__":
    raise SystemExit(main())
