"""Shared, explicit policy for the new MobileVLA development comparison.

The frozen 006-002 candidate and comparison006-002 helpers are deliberately not
modified. Token IDs are raw generated IDs, including the terminal EOS. Text is
diagnostic because detokenization can differ between engines.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

EOS_IDS = [128009]
SOURCE_GENERATION_EOS_IDS = [128001, 128009]
DEFAULT_QUESTION = "Describe the visible scene, then state a suitable navigation action in <answer>...</answer>."


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generation_policy(source, tokenizer, max_new_tokens):
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 128:
        raise ValueError("Comparison requires 1-128 new tokens")
    source = Path(source)
    path = source / "llm/generation_config.json"
    source_config = json.loads(path.read_text(encoding="utf-8"))
    if source_config.get("eos_token_id") != SOURCE_GENERATION_EOS_IDS:
        raise ValueError("Source generation EOS differs from the pinned RL checkpoint")
    if (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) != (128000, 128009, 128256):
        raise ValueError("Source tokenizer special-token identities differ")
    return {"mode": "greedy", "max_new_tokens": max_new_tokens,
            "publisher_sampling_source": "inference.py sample_prepared explicit tokenizer.eos_token_id",
            "eos_token_ids": EOS_IDS.copy(), "bos_token_id": 128000,
            "pad_token_id": 128256, "repetition_penalty": 1.0,
            "presence_penalty": 0.0, "frequency_penalty": 0.0,
            "min_new_tokens": 0, "terminal_eos_in_generated_token_ids": True,
            "source_generation_config_sha256": sha256(path)}


def native_parameters(policy):
    # The004 derived view maps root/text/generation EOS to publisher128009;
    # the installed native EOS probe must verify its effective stop set first.
    return {"temperature": 0, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
            "repetition_penalty": 1.0, "presence_penalty": 0.0,
            "frequency_penalty": 0.0, "min_tokens": 0,
            "max_tokens": policy["max_new_tokens"], "stop": [],
            "stop_token_ids": policy["eos_token_ids"].copy(), "ignore_eos": False,
            "return_token_ids": True, "skip_special_tokens": True,
            "spaces_between_special_tokens": False, "stream": False, "n": 1}


def token_list(value):
    # The image sentinel is an input-only positive surrogate, outside the LLM
    # output vocabulary. bool is intentionally not accepted as a token ID.
    return isinstance(value, list) and all(type(x) is int and 0 <= x < 128259 for x in value)


def validate_completion(tokens, finish_reason, stop_reason, policy, usage=None):
    problems = []
    maximum = policy["max_new_tokens"]
    if not token_list(tokens) or not 1 <= len(tokens) <= maximum:
        return ["missing_invalid_or_unbounded_generated_token_ids"]
    eos_positions = [i for i, token in enumerate(tokens) if token in policy["eos_token_ids"]]
    if eos_positions and eos_positions != [len(tokens) - 1]:
        problems.append("generation_continued_after_EOS")
    if finish_reason == "length":
        if len(tokens) != maximum:
            problems.append("length_finish_without_maximum_generated_tokens")
    elif finish_reason == "stop":
        if tokens[-1] not in policy["eos_token_ids"]:
            problems.append("stop_finish_without_terminal_EOS")
        if stop_reason is not None and (type(stop_reason) is not int or stop_reason != tokens[-1]):
            problems.append("stop_reason_does_not_match_terminal_token")
    else:
        problems.append("unexpected_finish_reason")
    if usage is not None:
        if not isinstance(usage, dict) or type(usage.get("completion_tokens")) is not int or usage["completion_tokens"] != len(tokens):
            problems.append("usage_completion_tokens_differs_from_raw_token_count")
    return problems


def first_divergence(expected, actual):
    for index, (left, right) in enumerate(zip(expected, actual)):
        if left != right:
            return {"index": index, "reference": left, "native": right}
    if len(expected) != len(actual):
        index = min(len(expected), len(actual))
        return {"index": index, "reference": expected[index] if index < len(expected) else None,
                "native": actual[index] if index < len(actual) else None}
    return None


def stop_boundary_coverage(record):
    tokens = record.get("generated_token_ids", [])
    valid = token_list(tokens) and bool(tokens)
    eos = valid and tokens[-1] == 128009 and record.get("finish_reason") == "stop"
    positions = [i for i, token in enumerate(tokens) if token == 128001] if valid else []
    crossed = any(i < len(tokens) - 1 for i in positions)
    return {"max_new_tokens": record.get("generation_policy", {}).get("max_new_tokens"),
            "generated_tokens": len(tokens) if valid else None,
            "token128001_positions": positions, "continued_after_nonpublisher128001": crossed,
            "publisher128009_terminal_EOS_observed": eos,
            "stop_boundary_exercised": crossed or eos}


def compare_records(native, reference):
    problems = []
    if native.get("schema") != "mobilevla-native-comparison006-v4" or reference.get("schema") != "mobilevla-hf-comparison006-v4":
        problems.append("requires_both_new_v4_helpers")
    if native.get("status") != "completed_comparable" or reference.get("status") != "completed_comparable":
        problems.append("one_or_both_runs_are_incomplete_or_not_comparable")
    for key in ("generation_policy", "source_prompt_token_ids", "expected_expanded_prompt_token_ids", "source_prompt"):
        if native.get(key) != reference.get(key) or native.get(key) is None:
            problems.append(key + "_differs_or_missing")
    native_images = [x.get("sha256") for x in native.get("images", [])]
    reference_images = [x.get("sha256") for x in reference.get("images", [])]
    if not native_images or native_images != reference_images:
        problems.append("ordered_image_bytes_differ_or_missing")
    actual = native.get("generated_token_ids")
    expected = reference.get("generated_token_ids")
    if not token_list(actual) or not token_list(expected):
        problems.append("raw_generated_token_ids_missing_or_invalid")
    policy = native.get("generation_policy")
    if isinstance(policy, dict) and set(("max_new_tokens", "eos_token_ids")) <= set(policy):
        for label, record in (("native", native), ("reference", reference)):
            found = validate_completion(record.get("generated_token_ids"), record.get("finish_reason"),
                                        record.get("stop_reason"), policy,
                                        record.get("usage", {}) if label == "native" else None)
            problems.extend(label + ":" + reason for reason in found)
    else:
        problems.append("generation_policy_missing_or_incomplete")
    if native.get("actual_prompt_token_ids") != native.get("expected_expanded_prompt_token_ids"):
        problems.append("native_actual_prompt_ids_differ_from_expected")
    comparable = not problems
    return {"schema": "mobilevla-development-token-comparison006-v4", "comparable": comparable,
            "contract_problems": problems,
            "raw_generated_token_ids_equal": expected == actual if comparable else None,
            "first_generated_divergence": first_divergence(expected, actual) if comparable else None,
            "native_generated_tokens": len(actual) if token_list(actual) else None,
            "reference_generated_tokens": len(expected) if token_list(expected) else None,
            "terminal_EOS_comparison": "included in raw token IDs on both sides",
            "native_stop_boundary_coverage": stop_boundary_coverage(native),
            "reference_stop_boundary_coverage": stop_boundary_coverage(reference),
            "official_accuracy_verified": False, "official_performance_verified": False,
            "scope": "Exact same-input development comparison; equality is not navigation accuracy or an official score"}
