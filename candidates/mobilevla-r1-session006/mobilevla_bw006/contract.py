"""Fixed MobileVLA v1 RGB contract. No model imports or environment mutation."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

REVISION = "0c757f3e9d1b95d8416b42e7089e1c04f3e6118a"
SOURCE_COMMIT = "9d98cce3b623dfddf479740d78869ec30364b75a"
IMAGE_TOKEN_ID = 128259  # Positive runtime equivalent of upstream -200 sentinel.
IMAGE_TOKEN = "<image>"
VISION_PATCHES = 27 * 27
PROJECTED_TOKENS = 14 * 14
DERIVED_SMALL_FILES = frozenset({
    "config.json", "tokenizer.json", "tokenizer_config.json",
    "mobilevla-chat-template.jinja", "model.safetensors.index.json",
    "preprocessor_config.json", "generation_config.json", "special_tokens_map.json",
})
SYSTEM_TEXT = (
    "You are a helpful language and vision assistant. "
    "You are able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language."
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def selected_image_tokens(patches: int) -> int:
    side = math.isqrt(patches)
    if side * side != patches or side == 0:
        raise ValueError("MobileVLA downsample requires a nonempty square patch grid")
    return ((side + 1) // 2) ** 2


def validate_checkpoint_config(root: Path) -> dict:
    """Reject a different model/disabled modality silently entering this adapter."""
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    llm = json.loads((root / "llm/config.json").read_text(encoding="utf-8"))
    vision = json.loads((root / "vision_tower/config.json").read_text(encoding="utf-8"))
    projector = json.loads((root / "mm_projector/config.json").read_text(encoding="utf-8"))
    exact = {
        "architectures": ["LlavaLlamaModel"], "model_type": "llava_llama",
        "mm_hidden_size": 1152, "hidden_size": 4096,
        "mm_vision_select_layer": -2, "mm_vision_select_feature": "cls_patch",
        "image_aspect_ratio": "resize", "s2": False,
        "mm_use_im_start_end": False, "mm_use_im_patch_token": False,
    }
    for key, expected in exact.items():
        if cfg.get(key) != expected:
            raise ValueError(f"MobileVLA top-level {key} differs from pinned RGB checkpoint")
    if any(cfg.get(k, False) for k in ("use_depth_tower", "use_point_tower")):
        raise ValueError("This adapter only implements the published RGB checkpoint")
    if cfg.get("auxiliary_weights_file"):
        raise ValueError("An auxiliary checkpoint requires a separate audited modality path")
    for key, expected in {"model_type": "llama", "hidden_size": 4096,
                          "num_hidden_layers": 32, "vocab_size": IMAGE_TOKEN_ID,
                          "num_attention_heads": 32, "num_key_value_heads": 8,
                          "max_position_embeddings": 8192}.items():
        if llm.get(key) != expected:
            raise ValueError(f"LLM {key} differs from fixed checkpoint")
    for key, expected in {"hidden_size": 1152, "image_size": 384, "patch_size": 14,
                          "num_hidden_layers": 27, "num_attention_heads": 16}.items():
        if vision.get(key) != expected:
            raise ValueError(f"SigLIP {key} differs from fixed checkpoint")
    if projector.get("mm_projector_type") != "mlp_downsample":
        raise ValueError("Only the actual 2x2 downsample projector is supported")
    return {"source": cfg, "llm": llm, "vision": vision, "projector": projector}


def map_weight_name(name: str) -> str:
    # The native default loader adds these two prefixes via secondary_weights.
    if name.startswith(("vision_tower.", "multi_modal_projector.")):
        return name
    if name.startswith(("model.", "lm_head.")):
        return "language_model." + name
    raise ValueError(f"Unexpected MobileVLA checkpoint parameter: {name}")


def tokenize_source_prompt(prompt, tokenizer, image_token_id=IMAGE_TOKEN_ID):
    """Publisher tokenizer_image_token semantics, including its BOS offset."""
    chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(IMAGE_TOKEN)]
    result = []
    offset = 0
    if chunks and chunks[0] and chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        result.append(chunks[0][0])
    for index, chunk in enumerate(chunks):
        if index:
            result.append(image_token_id)
        result.extend(chunk[offset:])
    return result


def chat_template() -> str:
    """Default system and separators match publisher inference.py llama_3 mode.

    API content must already be rendered as a string by vLLM. The caller supplies
    observation_question's labels and frame newlines for strict reference tests.
    """
    # Source Conversation emits the system followed by eot, and assistant header
    # without an eot for generation. Use the exact bytes, without trim filters.
    system = json.dumps(SYSTEM_TEXT)
    return (
        "{{ bos_token }}"
        "{% if messages and messages[0]['role'] == 'system' %}"
        "{% set start = 1 %}{% set sys = messages[0]['content'] %}"
        "{% else %}{% set start = 0 %}{% set sys = " + system + " %}{% endif %}"
        "{{ '<|start_header_id|>system<|end_header_id|>\\n\\n' + sys + '<|eot_id|>' }}"
        "{% for message in messages[start:] %}"
        "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] }}"
        "{% if loop.last and not add_generation_prompt %}{{ '<|end_of_text|>' }}"
        "{% else %}{{ '<|eot_id|>' }}{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}"
    )
