"""CPU comparisons against pinned source, no model weights or GPU required."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "candidates/mobilevla-r1-session006"
sys.path.insert(0, str(CANDIDATE))
from mobilevla_bw006 import ARCHITECTURE
from mobilevla_bw006.contract import (DERIVED_SMALL_FILES, IMAGE_TOKEN_ID, chat_template,
                                     map_weight_name, selected_image_tokens,
                                     tokenize_source_prompt, validate_checkpoint_config)
from mobilevla_bw006.prepare import read_safetensors_header, seal_derived_small_files


class DerivedSealTests(unittest.TestCase):
    """Small synthetic metadata tests; they are not model loading evidence."""
    def setUp(self):
        from mobilevla_bw006 import runtime
        from mobilevla_bw006.contract import sha256_file
        self.runtime = runtime
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name).resolve()
        self.source = self.base / "original"
        self.model = self.base / "derived"
        self.model.mkdir()
        plan = json.loads((CANDIDATE / "rl-entry-plan.json").read_text())
        sealed = {}
        for item in plan["entries"]:
            name = item["path"].removeprefix("weight/rl/")
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("synthetic metadata-test bytes for " + name).encode())
            sealed[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            if name.startswith("llm/") and name.endswith(".safetensors"):
                os.link(path, self.model / Path(name).name)
        for name in DERIVED_SMALL_FILES:
            (self.model / name).write_text("fixture bytes for " + name)
        (self.model / "config.json").write_text(json.dumps({
            "architectures": [ARCHITECTURE], "image_token_index": IMAGE_TOKEN_ID,
            "mobilevla_source_checkpoint": str(self.source), "mobilevla_adapter_revision": "007-eos004",
            "eos_token_id": 128009, "text_config": {"eos_token_id": 128009},
        }))
        (self.model / "generation_config.json").write_text(json.dumps({"eos_token_id": 128009}))
        self.record = {"schema": "mobilevla-derived-hf-view-007-eos004-v1", "adapter_revision": "007-eos004",
                       "adapter_source_files": {p.name: sha256_file(p) for p in sorted(Path(self.runtime.__file__).parent.glob("*.py"))}, "derived": str(self.model),
                       "source": str(self.source), "source_unchanged": True, "source_files": sealed,
                       "publisher_eos_mapping": {"derived_root_text_and_generation_eos_token_id": 128009},
                       "derived_small_files": seal_derived_small_files(self.model)}
        self.save_receipt()

    def save_receipt(self):
        (self.model / "prepare-receipt.json").write_text(json.dumps(self.record))

    def tearDown(self):
        self.temp.cleanup()

    def test_unchanged_source_and_eight_derived_files_pass(self):
        result = self.runtime.checkpoint(self.model, hash_weights=True)
        self.assertTrue(result["complete"])
        self.assertEqual({x["path"] for x in result["derived_small_files"]}, DERIVED_SMALL_FILES)

    def test_derived_tokenizer_tamper_rejected_with_source_unchanged(self):
        path = self.model / "tokenizer.json"
        original = path.read_bytes()
        path.write_bytes(bytes([original[0] ^ 1]) + original[1:])  # Same length.
        result = self.runtime.checkpoint(self.model, hash_weights=True)
        self.assertFalse(result["complete"])
        self.assertEqual(result["reason"], "derived small file no longer matches prepare receipt")

    def test_additional_generation_EOS_is_rejected_even_if_receipt_is_resealed(self):
        path = self.model / "generation_config.json"
        path.write_text(json.dumps({"eos_token_id": [128001, 128009]}))
        self.record["derived_small_files"] = seal_derived_small_files(self.model)
        self.save_receipt()
        with self.assertRaisesRegex(ValueError, "EOS must match publisher"):
            self.runtime.checkpoint(self.model, hash_weights=True)

    def test_chat_template_tamper_also_checked_without_weight_hashing(self):
        (self.model / "mobilevla-chat-template.jinja").write_text("changed template")
        result = self.runtime.checkpoint(self.model, hash_weights=False)
        self.assertFalse(result["complete"])
        self.assertTrue(any(not x["identity_ok"] for x in result["derived_small_files"]))

    def test_derived_source_path_edit_rejected(self):
        path = self.model / "config.json"
        config = json.loads(path.read_text())
        config["mobilevla_source_checkpoint"] += "-different"
        path.write_text(json.dumps(config))
        self.assertFalse(self.runtime.checkpoint(self.model, hash_weights=True)["complete"])

    def test_receipt_missing_derived_entry_rejected(self):
        del self.record["derived_small_files"]["preprocessor_config.json"]
        self.save_receipt()
        with self.assertRaisesRegex(ValueError, "all eight"):
            self.runtime.checkpoint(self.model, hash_weights=True)

    def test_invalid_null_derived_digest_rejected(self):
        self.record["derived_small_files"]["preprocessor_config.json"]["sha256"] = None
        self.save_receipt()
        with self.assertRaisesRegex(ValueError, "Invalid generated-file seal"):
            self.runtime.checkpoint(self.model, hash_weights=False)

    def test_derived_preparation_path_drift_rejected(self):
        self.record["derived"] = str(self.base / "other")
        self.save_receipt()
        with self.assertRaisesRegex(ValueError, "preparation identity"):
            self.runtime.checkpoint(self.model, hash_weights=True)

    def test_derived_file_symlink_rejected(self):
        path = self.model / "tokenizer.json"
        path.unlink()
        target = self.base / "same-tokenizer-bytes"
        target.write_text("fixture bytes for tokenizer.json")
        try:
            path.symlink_to(target)
        except OSError:
            self.skipTest("Windows symlink privilege unavailable")
        result = self.runtime.checkpoint(self.model, hash_weights=False)
        self.assertFalse(result["complete"])


def source_items(relative, names, namespace):
    tree = ast.parse((CANDIDATE / "reference" / relative).read_text(encoding="utf-8"))
    wanted = [x for x in tree.body if isinstance(x, (ast.ClassDef, ast.FunctionDef)) and x.name in names]
    if {x.name for x in wanted} != set(names):
        raise AssertionError("Pinned source definitions missing")
    exec(compile(ast.Module(body=wanted, type_ignores=[]), relative, "exec"), namespace)
    return namespace


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "source"
        shutil.copytree(CANDIDATE / "reference/rl-metadata", self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_real_pinned_checkpoint_contract(self):
        cfg = validate_checkpoint_config(self.root)
        self.assertEqual(cfg["llm"]["vocab_size"], IMAGE_TOKEN_ID)
        self.assertEqual(selected_image_tokens(729), 196)

    def mutate(self, key, value):
        path = self.root / "config.json"
        cfg = json.loads(path.read_text())
        cfg[key] = value
        path.write_text(json.dumps(cfg))
        with self.assertRaises(ValueError):
            validate_checkpoint_config(self.root)

    def test_s2_is_not_same_model(self):
        self.mutate("s2", True)

    def test_depth_not_implemented(self):
        self.mutate("use_depth_tower", True)

    def test_feature_layer_not_silently_changed(self):
        self.mutate("mm_vision_select_layer", -1)

    def test_no_cls_is_dropped_from_siglip(self):
        self.mutate("mm_vision_select_feature", "patch")

    def test_weight_prefix_sources(self):
        self.assertEqual(map_weight_name("model.layers.0.self_attn.q_proj.weight"),
                         "language_model.model.layers.0.self_attn.q_proj.weight")
        self.assertEqual(map_weight_name("lm_head.weight"), "language_model.lm_head.weight")
        for key in ("vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight",
                    "multi_modal_projector.layers.2.weight"):
            self.assertEqual(map_weight_name(key), key)
        with self.assertRaises(ValueError):
            map_weight_name("random.weight")

    def test_nonsquare_patch_counts_rejected(self):
        for count in (0, 728, 730):
            with self.assertRaises(ValueError):
                selected_image_tokens(count)

    def test_plugin_registered_as_lazy_custom_implementation(self):
        import importlib.metadata
        entries = list(importlib.metadata.entry_points(group="vllm.general_plugins"))
        ours = [e for e in entries if e.name == "mobilevla_bw006"]
        self.assertEqual(len(ours), 1)
        self.assertEqual(ours[0].value, "mobilevla_bw006:register")
        self.assertEqual(ARCHITECTURE, "MobileVLAForConditionalGeneration006")

    def test_safetensors_header_truncated_rejected(self):
        path = Path(self.temp.name) / "bad.safetensors"
        path.write_bytes(b"\xff" * 8)
        with self.assertRaises(ValueError):
            read_safetensors_header(path)


class ProjectorSourceParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from torch import nn
        from mobilevla_bw006.projector import DownSampleBlock, MobileVLAProjector
        cls.torch, cls.nn = torch, nn
        cls.adapted_downsample = DownSampleBlock
        cls.adapted_projector = MobileVLAProjector

        class PreTrainedModel(nn.Module):
            def __init__(self, config):
                super().__init__()
        namespace = source_items("llava/model/multimodal_projector/base_projector.py",
                                 ["DownSampleBlock", "MultimodalProjector"],
                                 {"torch": torch, "nn": nn, "PreTrainedModel": PreTrainedModel,
                                  "MultimodalProjectorConfig": SimpleNamespace,
                                  "PretrainedConfig": SimpleNamespace})
        cls.source_downsample = namespace["DownSampleBlock"]
        cls.source_projector = namespace["MultimodalProjector"]

    def compare_grid(self, frames, side, channels):
        torch = self.torch
        x = torch.arange(frames * side * side * channels, dtype=torch.float32).reshape(frames, side * side, channels)
        actual = self.adapted_downsample()(x)
        expected = self.source_downsample()(x)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(list(actual.shape), [frames, ((side + 1) // 2) ** 2, 4 * channels])
        return actual

    def test_even_grid_actual_source_equivalence(self):
        self.compare_grid(2, 4, 3)

    def test_odd_grid_padding_actual_source_equivalence(self):
        self.compare_grid(2, 3, 3)

    def test_real_27_grid_eight_frame_order(self):
        actual = self.compare_grid(8, 27, 2)
        self.assertEqual(actual.shape[1], 196)
        # Distinct frame offsets must remain distinct and in original order.
        self.assertEqual(actual[:, 0, 0].tolist(), [float(i * 729 * 2) for i in range(8)])

    def test_explicit_flattening_order(self):
        torch = self.torch
        x = torch.arange(16, dtype=torch.float32).reshape(1, 16, 1)
        self.assertEqual(self.adapted_downsample()(x).tolist(),
                         [[[0., 1., 4., 5.], [8., 9., 12., 13.],
                           [2., 3., 6., 7.], [10., 11., 14., 15.]]])

    def test_full_projector_source_numerical_equivalence(self):
        torch = self.torch
        torch.manual_seed(1306)
        source = self.source_projector(SimpleNamespace(mm_projector_type="mlp_downsample"),
                                       SimpleNamespace(mm_hidden_size=2, hidden_size=3)).eval()
        adapted = self.adapted_projector(2, 3).eval()
        adapted.load_state_dict(source.state_dict(), strict=True)
        x = torch.randn(8, 729, 2)
        with torch.inference_mode():
            self.assertTrue(torch.equal(source(x), adapted(x)))
            permutation = torch.tensor([7, 0, 4, 1, 6, 3, 5, 2])
            self.assertTrue(torch.equal(adapted(x[permutation]), adapted(x)[permutation]))


class TokenizerAndTemplateSourceParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import PreTrainedTokenizerFast
        import torch
        path = CANDIDATE / "reference/rl-metadata/llm/tokenizer.json"
        if not path.is_file():
            raise unittest.SkipTest("Bounded publisher tokenizer Range has not arrived")
        cfg = json.loads((path.parent / "tokenizer_config.json").read_text())
        cls.original = PreTrainedTokenizerFast(tokenizer_file=str(path),
                                             bos_token=cfg["bos_token"], eos_token=cfg["eos_token"],
                                             pad_token=cfg["pad_token"])
        ns = source_items("llava/mm_utils.py", ["tokenizer_image_token"],
                          {"torch": torch, "IMAGE_TOKEN_INDEX": -200})
        cls.source_tokenize = staticmethod(ns["tokenizer_image_token"])
        # Execute only the actual enum/dataclass/template AST nodes. Do not import
        # unrelated llava logging/model/train dependencies into the CPU test.
        import dataclasses
        import enum
        import types
        import typing
        module = types.ModuleType("mobilevla_source_conversation006")
        module.__dict__.update(dataclasses=dataclasses, Enum=enum.Enum,
                               auto=enum.auto, List=typing.List)
        sys.modules[module.__name__] = module
        tree = ast.parse((CANDIDATE / "reference/llava/conversation.py").read_text())
        nodes = [n for n in tree.body if
                 isinstance(n, ast.ClassDef) and n.name in ("SeparatorStyle", "Conversation")
                 or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "llama_3_chat" for t in n.targets)]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "source-conversation", "exec"), module.__dict__)
        cls.conversation = module.llama_3_chat

    def source_prompt(self, frames, history=False):
        conv = self.conversation.copy()
        if history:
            conv.append_message(conv.roles[0], "Earlier observation.")
            conv.append_message(conv.roles[1], "Earlier answer.")
        question = "RGB observations:\n" + "<image>\n" * frames + "Turn left at the door."
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt(), question

    def test_real_tokenizer_source_image_sentinel_equivalence(self):
        for frames in (1, 8):
            prompt, _ = self.source_prompt(frames)
            source = self.source_tokenize(prompt, self.original)
            adapted = tokenize_source_prompt(prompt, self.original)
            self.assertEqual([IMAGE_TOKEN_ID if x == -200 else x for x in source], adapted)
            self.assertEqual(adapted.count(IMAGE_TOKEN_ID), frames)

    def test_source_chat_single_and_multiturn_exact_text(self):
        import jinja2
        template = jinja2.Template(chat_template())
        for history in (False, True):
            source, question = self.source_prompt(8, history)
            messages = []
            if history:
                messages.extend([{"role": "user", "content": "Earlier observation."},
                                 {"role": "assistant", "content": "Earlier answer."}])
            messages.append({"role": "user", "content": question})
            actual = template.render(messages=messages, add_generation_prompt=True,
                                     bos_token="<|begin_of_text|>")
            self.assertEqual(actual, source)

    def test_derived_sentinel_preserves_all_other_token_ids(self):
        from transformers import PreTrainedTokenizerFast
        tokenizer = json.loads((CANDIDATE / "reference/rl-metadata/llm/tokenizer.json").read_text())
        tokenizer["added_tokens"].append({"id": IMAGE_TOKEN_ID, "content": "<image>",
                                          "single_word": False, "lstrip": False, "rstrip": False,
                                          "normalized": False, "special": True})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tokenizer.json"
            path.write_text(json.dumps(tokenizer))
            derived = PreTrainedTokenizerFast(tokenizer_file=str(path),
                                             bos_token="<|begin_of_text|>",
                                             eos_token="<|eot_id|>", pad_token="[PAD]")
            self.assertEqual(len(derived), 128260)
            for text in ("Move forward 25 centimeters.", "向左转30度。", "<think>door</think><answer>stop</answer>"):
                self.assertEqual(self.original.encode(text), derived.encode(text))
            for frames in (1, 8):
                prompt, _ = self.source_prompt(frames)
                reference = tokenize_source_prompt(prompt, self.original)
                # Real fast-tokenizer sentinel, as used by the OpenAI renderer.
                self.assertEqual(derived.encode(prompt), reference)

    def test_siglip_preprocess_fixed_pixels_source_equivalence(self):
        from transformers import BatchFeature, SiglipImageProcessor
        from PIL import Image
        import torch
        processor = SiglipImageProcessor.from_pretrained(
            str(CANDIDATE / "reference/rl-metadata/vision_tower"), local_files_only=True)
        ns = source_items("llava/mm_utils.py", ["process_image"],
                          {"Image": Image, "os": __import__("os")})
        # Non-square and non-RGB inputs expose resize/channel-order regressions.
        images = [Image.new("RGB", (53, 29), (193, 7, 41)),
                  Image.new("L", (17, 31), 113),
                  Image.new("RGBA", (38, 23), (7, 151, 241, 127))]
        config = SimpleNamespace(image_processor=processor, image_aspect_ratio="resize")
        expected = torch.stack([ns["process_image"](image, config, None) for image in images])
        # Invoke the actual adapter class method while replacing only its vLLM
        # base class, since no native vLLM/DCU runtime is installed on Windows.
        tree = ast.parse((CANDIDATE / "mobilevla_bw006/backend.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MobileVLAMultiModalProcessor")
        namespace = {"LlavaMultiModalProcessor": object, "torch": torch,
                     "BatchFeature": BatchFeature, "tokenize_source_prompt": tokenize_source_prompt}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "actual-adapter-processor", "exec"), namespace)
        adapter = namespace["MobileVLAMultiModalProcessor"]()
        adapter.info = SimpleNamespace(get_tokenizer=lambda: self.original,
                                       get_hf_processor=lambda: processor)
        prompt = "<image>\n<image>\n<image>\nDescribe the frames."
        result = adapter._call_hf_processor(prompt, {"images": images}, {}, {})
        actual = result["pixel_values"]
        self.assertEqual(list(actual.shape), [3, 3, 384, 384])
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(result["input_ids"][0].tolist().count(IMAGE_TOKEN_ID), 3)
        self.assertFalse(adapter._hf_processor_applies_updates())
        with self.assertRaises(ValueError):
            adapter._call_hf_processor(prompt, {"images": images}, {"do_resize": False}, {})
        with self.assertRaises(ValueError):
            adapter._call_hf_processor(prompt, {"images": images}, {}, {"truncation": True})

    def test_native_prompt_replacement_is_exact_196_per_frame(self):
        import torch
        source = ROOT / "evidence/mobilevla-preflight-20261023/framework-source/vllm-v0.18.1-llava.py"
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BaseLlavaMultiModalProcessor")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_get_prompt_updates")
        class ImageProcessorItems:
            def get_image_size(self, index):
                return SimpleNamespace(width=37 + index, height=91 - index)
        class ImageEmbeddingItems:
            pass
        namespace = {"ImageProcessorItems": ImageProcessorItems,
                     "ImageEmbeddingItems": ImageEmbeddingItems,
                     "PromptReplacement": lambda **kw: SimpleNamespace(**kw)}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "native-vllm-prompt-replacement", "exec"), namespace)
        info = SimpleNamespace(get_hf_config=lambda: SimpleNamespace(image_token_index=IMAGE_TOKEN_ID),
                               get_num_image_tokens=lambda **kw: 196)
        items = SimpleNamespace(get_items=lambda *a: ImageProcessorItems())
        updates = namespace["_get_prompt_updates"](SimpleNamespace(info=info), items, {}, None)
        self.assertEqual(updates[0].target, [IMAGE_TOKEN_ID])
        for frame in range(8):
            self.assertEqual(updates[0].replacement(frame), [IMAGE_TOKEN_ID] * 196)


if __name__ == "__main__":
    unittest.main()
