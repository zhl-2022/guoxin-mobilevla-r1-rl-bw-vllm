"""Dedicated vLLM 0.18.1 RGB adapter, not an architecture-name-only override.

Native Llama3/SigLIP attention and scheduling remain in vLLM. This module supplies
the publisher's downsample projector, frame/token contract, preprocessing,
component weight sources and explicit prefix mapping. New adaptation copyright
reserved; upstream-derived pieces retain Apache-2.0 notices.
"""
from functools import cached_property
from pathlib import Path

import torch
from transformers import BatchFeature, SiglipImageProcessor
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.models.llava import (
    LlavaDummyInputsBuilder,
    LlavaForConditionalGeneration,
    LlavaMultiModalProcessor,
    LlavaProcessingInfo,
)
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.multimodal import MULTIMODAL_REGISTRY

from .contract import (IMAGE_TOKEN, IMAGE_TOKEN_ID, PROJECTED_TOKENS,
                       VISION_PATCHES, map_weight_name, tokenize_source_prompt)
from .projector import MobileVLAProjector


class MobileVLAProcessingInfo(LlavaProcessingInfo):
    @cached_property
    def _processor(self):
        cfg = self.get_hf_config()
        processor = SiglipImageProcessor.from_pretrained(
            str(Path(cfg.mobilevla_source_checkpoint) / "vision_tower"),
            local_files_only=True,
        )
        # Dummy builder only needs .image_token; processor call is overridden.
        processor.image_token = IMAGE_TOKEN
        return processor

    def get_hf_processor(self, **kwargs):
        if kwargs:
            raise ValueError("MobileVLA fixed image preprocessing disallows runtime overrides")
        return self._processor

    def get_num_image_tokens(self, *, image_width, image_height):
        # All frames are independently resized to384x384, then27^2->14^2.
        return PROJECTED_TOKENS


class MobileVLAMultiModalProcessor(LlavaMultiModalProcessor):
    def _hf_processor_applies_updates(self, *args, **kwargs):
        # Our call leaves one sentinel/frame. Native PromptReplacement expands it.
        return False

    def _apply_hf_processor_tokens_only(self, prompt_tokens):
        # OpenAI chat is rendered/tokenized before this processor. The base
        # tokens-only path returns those IDs unchanged, bypassing the publisher
        # tokenizer_image_token semantics used by our text path. Decode without
        # cleanup, then apply the same source tokenizer to the corresponding
        # text. In particular, the checkpoint's automatic BOS and Conversation's
        # literal BOS are both preserved; no returned/generated IDs are trimmed.
        tokenizer = self.info.get_tokenizer()
        prompt = tokenizer.decode(prompt_tokens, skip_special_tokens=False,
                                  clean_up_tokenization_spaces=False)
        return tokenize_source_prompt(prompt, tokenizer)

    def _call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        if mm_kwargs:
            raise ValueError("Fixed MobileVLA RGB preprocessing cannot be overridden")
        unknown = set(mm_data) - {"images"}
        if unknown:
            raise ValueError(f"Only RGB images are supported, got {sorted(unknown)}")
        tokenizer = self.info.get_tokenizer()
        ids = tokenize_source_prompt(prompt, tokenizer)
        features = BatchFeature({"input_ids": torch.tensor([ids], dtype=torch.long)})
        images = mm_data.get("images")
        if images is not None:
            if not isinstance(images, (list, tuple)):
                images = [images]
            # Publisher process_image explicitly converts and resizes each frame
            # using PIL default BICUBIC before SiglipImageProcessor.preprocess.
            ordered = [image.convert("RGB").resize((384, 384)) for image in images]
            pixels = self.info.get_hf_processor().preprocess(
                ordered, return_tensors="pt"
            )["pixel_values"]
            if list(pixels.shape) != [len(images), 3, 384, 384]:
                raise ValueError("Unexpected fixed RGB pixel shape")
            features["pixel_values"] = pixels
        # tok_kwargs are recorded by vLLM; no truncation is silently applied here.
        # Reject actual truncation instead of splitting an observation's features.
        if tok_kwargs.get("truncation"):
            raise ValueError("MobileVLA observations cannot be silently truncated")
        return features


@MULTIMODAL_REGISTRY.register_processor(
    MobileVLAMultiModalProcessor,
    info=MobileVLAProcessingInfo,
    dummy_inputs=LlavaDummyInputsBuilder,
)
class MobileVLAForConditionalGeneration006(LlavaForConditionalGeneration):
    def __init__(self, *, vllm_config, prefix=""):
        if vllm_config.parallel_config.tensor_parallel_size != 1:
            raise ValueError("Initial MobileVLA adapter is validated only for TP1")
        if vllm_config.quant_config is not None:
            raise ValueError("MobileVLA RL adapter currently requires original BF16 weights")
        config = vllm_config.model_config.hf_config
        if config.image_token_index != IMAGE_TOKEN_ID:
            raise ValueError("Derived sentinel identity mismatch")
        if config.vision_feature_layer != -2 or config.vision_feature_select_strategy != "full":
            raise ValueError("MobileVLA requires raw SigLIP hidden_states[-2], all patches")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.multi_modal_projector = MobileVLAProjector()
        source = Path(config.mobilevla_source_checkpoint)
        self.secondary_weights = [
            DefaultModelLoader.Source(
                str(source / "vision_tower"), revision=None,
                prefix="vision_tower.", fall_back_to_pt=False,
            ),
            DefaultModelLoader.Source(
                str(source / "mm_projector"), revision=None,
                prefix="multi_modal_projector.", fall_back_to_pt=False,
            ),
        ]

    def load_weights(self, weights):
        # Strict default loader still requires every actual model parameter loaded.
        mapped = ((map_weight_name(name), tensor) for name, tensor in weights)
        return AutoWeightsLoader(self).load_weights(mapped)

    def get_num_mm_encoder_tokens(self, num_image_tokens):
        if num_image_tokens != PROJECTED_TOKENS:
            raise ValueError("Unexpected projected image token count")
        return VISION_PATCHES

    def get_num_mm_connector_tokens(self, num_vision_tokens):
        if num_vision_tokens != VISION_PATCHES:
            raise ValueError("Unexpected SigLIP patch count")
        return PROJECTED_TOKENS
