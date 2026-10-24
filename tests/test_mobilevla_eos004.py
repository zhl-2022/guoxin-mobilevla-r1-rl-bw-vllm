"""Publisher AST, real checkpoint tokenizer and real vLLM token-path regression."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import Mapping
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REVISION = ROOT / "revisions/mobilevla-eos004"
if not REVISION.exists():
    REVISION = ROOT
PUBLISHER = ROOT / "evidence/mobilevla-preflight-20261023/source"
sys.path.insert(0, str(REVISION / "candidates/mobilevla-r1-session006"))
sys.path.insert(0, str(ROOT / "scripts"))
from mobilevla_bw006.contract import tokenize_source_prompt, IMAGE_TOKEN_ID
from mobilevla_bw006.reference_rgb import source_prompt
from mobilevla_compare_policy006_004 import generation_policy, native_parameters, validate_completion, DEFAULT_QUESTION


class SourceTokenPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The installed optional sklearn/scipy binaries are incompatible with
        # NumPy in this host. Disable only their availability flags in this CPU
        # test process so AutoTokenizer can be imported; no package/file edits.
        import transformers.utils.import_utils as flags
        flags._sklearn_available = False
        flags._scipy_available = False
        from transformers import AutoTokenizer, PreTrainedTokenizerFast
        cls.source = ROOT / "candidates/mobilevla-r1-session006/reference/rl-metadata"
        cls.tokenizer = AutoTokenizer.from_pretrained(str(cls.source / "llm"), local_files_only=True,
                                                      use_fast=False, legacy=False, padding_side="right")
        cls.tmp = tempfile.TemporaryDirectory()
        derived = json.loads((cls.source / "llm/tokenizer.json").read_text(encoding="utf-8"))
        derived["added_tokens"].append({"id": IMAGE_TOKEN_ID, "content": "<image>", "single_word": False,
                                        "lstrip": False, "rstrip": False, "normalized": False, "special": True})
        path = Path(cls.tmp.name) / "tokenizer.json"
        path.write_text(json.dumps(derived), encoding="utf-8")
        cls.derived = PreTrainedTokenizerFast(tokenizer_file=str(path), bos_token="<|begin_of_text|>",
                                              eos_token="<|eot_id|>", pad_token="[PAD]")
        source_method = next(n for n in ast.parse((PUBLISHER / "llava/mm_utils.py").read_text(encoding="utf-8")).body
                             if isinstance(n, ast.FunctionDef) and n.name == "tokenizer_image_token")
        namespace = {"IMAGE_TOKEN_INDEX": -200}
        exec(compile(ast.Module(body=[source_method], type_ignores=[]), "publisher-tokenizer", "exec"), namespace)
        cls.publisher_tokenize = staticmethod(namespace["tokenizer_image_token"])
        backend = REVISION / "candidates/mobilevla-r1-session006/mobilevla_bw006/backend.py"
        node = next(n for n in ast.parse(backend.read_text(encoding="utf-8")).body
                    if isinstance(n, ast.ClassDef) and n.name == "MobileVLAMultiModalProcessor")
        namespace = {"LlavaMultiModalProcessor": object, "tokenize_source_prompt": tokenize_source_prompt}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(backend), "exec"), namespace)
        cls.adapter = namespace[node.name]()
        cls.adapter.info = SimpleNamespace(get_tokenizer=lambda: cls.derived)
        cls.adapter._apply_hf_processor_mm_only = lambda **kw: {}
        framework = ROOT / "candidates/mobilevla-r1-session006/framework-source/vllm-multimodal-processing-processor.py"
        tree = ast.parse(framework.read_text(encoding="utf-8"))
        method = next(n for owner in tree.body if isinstance(owner, ast.ClassDef) for n in owner.body
                      if isinstance(n, ast.FunctionDef) and n.name == "_apply_hf_processor_main")
        namespace = {"Mapping": Mapping, "MultiModalDataItems": object, "BatchFeature": dict}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(framework), "exec"), namespace)
        cls.real_native_main = staticmethod(namespace[method.name])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_publisher_loader_source_and_files_match_fixed_Git_lock(self):
        lock = json.loads((PUBLISHER.parent / "source-file-lock.json").read_text(encoding="utf-8"))
        names = ["inference.py", "llava/model/llava_arch.py", "llava/model/language_model/builder.py", "llava/mm_utils.py", "llava/conversation.py"]
        for name in names:
            row = next(x for x in lock["files"] if x["path"] == name)
            self.assertEqual(hashlib.sha256((PUBLISHER / name).read_bytes()).hexdigest(), row["sha256"])
        builder = ast.parse((PUBLISHER / "llava/model/language_model/builder.py").read_text(encoding="utf-8"))
        function = next(n for n in builder.body if isinstance(n, ast.FunctionDef) and n.name == "build_llm_and_tokenizer")
        namespace = {"Tuple": tuple, "PreTrainedModel": object, "PreTrainedTokenizer": object,
                     "PretrainedConfig": object, "AutoConfig": SimpleNamespace(from_pretrained=lambda p: SimpleNamespace(architectures=["LlamaForCausalLM"])),
                     "AutoModelForCausalLM": SimpleNamespace(from_pretrained=lambda *a, **kw: SimpleNamespace(config=SimpleNamespace(hidden_size=4096))),
                     "has_tokenizer": lambda p: True, "osp": __import__("os").path}
        calls = []
        namespace["AutoTokenizer"] = SimpleNamespace(from_pretrained=lambda *a, **kw: calls.append(kw) or self.tokenizer)
        exec(compile(ast.Module(body=[function], type_ignores=[]), "publisher-builder", "exec"), namespace)
        namespace[function.name](str(self.source / "llm"), SimpleNamespace(model_dtype="object"))
        self.assertFalse(calls[0]["use_fast"])
        self.assertFalse(calls[0]["legacy"])

    def test_publisher_actual_tokenizer_preserves_two_BOS(self):
        self.assertEqual(type(self.tokenizer).__name__, "PreTrainedTokenizerFast")
        self.assertEqual(self.tokenizer("<|begin_of_text|>Hello").input_ids[:2], [128000, 128000])
        ids = self.publisher_tokenize(source_prompt(DEFAULT_QUESTION, 1), self.tokenizer)
        self.assertEqual(ids[:2], [128000, 128000])
        self.assertEqual(len(ids), 74)

    def test_real_native_tokens_main_path_matches_publisher_one_and_eight_frames(self):
        for frames in (1, 8):
            with self.subTest(frames=frames):
                prompt = source_prompt(DEFAULT_QUESTION, frames)
                rendered_ids = self.derived.encode(prompt, add_special_tokens=False)
                self.assertEqual(rendered_ids[:2], [128000, 128006])
                actual, _, applied = self.real_native_main(self.adapter, rendered_ids, None, {}, {}, enable_hf_prompt_update=True)
                expected = [IMAGE_TOKEN_ID if x == -200 else x for x in self.publisher_tokenize(prompt, self.tokenizer)]
                self.assertEqual(actual, expected)
                self.assertEqual(actual[:2], [128000, 128000])
                self.assertEqual(actual.count(IMAGE_TOKEN_ID), frames)
                self.assertFalse(applied)

    def test_failed_real_HTTP_raw_is_reproduced_and_new_path_fixes_upstream_input(self):
        path = ROOT / "dist/000-session007-mobile-comparison001.zip"
        with zipfile.ZipFile(path) as archive:
            record = json.loads(archive.read("comparison001/native-1.json"))
        self.assertEqual(len(record["actual_prompt_token_ids"]), 268)
        self.assertEqual(len(record["expected_expanded_prompt_token_ids"]), 269)
        prompt = record["source_prompt"]
        incoming = self.derived.encode(prompt, add_special_tokens=False)
        corrected = self.adapter._apply_hf_processor_tokens_only(incoming)
        expanded = []
        for token in corrected:
            expanded.extend([IMAGE_TOKEN_ID] * 196 if token == IMAGE_TOKEN_ID else [token])
        self.assertEqual(expanded, record["expected_expanded_prompt_token_ids"])
        self.assertNotEqual(expanded, record["actual_prompt_token_ids"])

    def test_publisher_sampling_EOS_is_single_and_invalid_stop_is_rejected(self):
        policy = generation_policy(self.source, self.tokenizer, 32)
        self.assertEqual(policy["eos_token_ids"], [128009])
        self.assertEqual(native_parameters(policy)["stop_token_ids"], [128009])
        self.assertEqual(validate_completion([31, 128001], "length", None, {**policy, "max_new_tokens": 2}), [])
        self.assertIn("stop_finish_without_terminal_EOS", validate_completion([31, 128001], "stop", 128001, policy))
        text = (REVISION / "candidates/mobilevla-r1-session006/mobilevla_bw006/runtime.py").read_text(encoding="utf-8")
        self.assertIn('"--generation-config", "vllm"', text)

    def test_old_derived_receipt_and_unbound_source_are_rejected_before_loading(self):
        from mobilevla_bw006 import ARCHITECTURE, runtime
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            (model / "config.json").write_text(json.dumps({"architectures": [ARCHITECTURE],
                "image_token_index": IMAGE_TOKEN_ID, "mobilevla_adapter_revision": "007-eos004",
                "eos_token_id": 128009, "text_config": {"eos_token_id": 128009}}))
            (model / "generation_config.json").write_text(json.dumps({"eos_token_id": 128009}))
            receipt = {"schema": "mobilevla-derived-hf-view-006-v2", "source_unchanged": True}
            (model / "prepare-receipt.json").write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):
                runtime.checkpoint(model)
            receipt.update(schema="mobilevla-derived-hf-view-007-eos004-v1", adapter_revision="007-eos004",
                           adapter_source_files={"backend.py": "0" * 64},
                           publisher_eos_mapping={"derived_root_text_and_generation_eos_token_id": 128009})
            (model / "prepare-receipt.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "adapter source identities differ"):
                runtime.checkpoint(model)


class EOSMetadataRegressionTests(unittest.TestCase):
    def test_mapping_changes_only_derived_eos_and_preserves_source_dicts(self):
        from mobilevla_bw006.prepare import publisher_eos_configs
        source = ROOT / "candidates/mobilevla-r1-session006/reference/rl-metadata/llm"
        text = json.loads((source / "config.json").read_text())
        generation = json.loads((source / "generation_config.json").read_text())
        before = json.dumps([text, generation], sort_keys=True)
        derived_text, derived_generation = publisher_eos_configs(text, generation)
        self.assertEqual(json.dumps([text, generation], sort_keys=True), before)
        self.assertEqual(derived_text, {**text, "eos_token_id": 128009})
        self.assertEqual(derived_generation, {**generation, "eos_token_id": 128009})
        with self.assertRaises(ValueError):
            publisher_eos_configs({**text, "eos_token_id": 0}, generation)

    def test_actual_vllm_AST_reproduces_vllm_mode_EOS_leak_and_004_fix(self):
        folder = ROOT / "evidence/session007/mobile-eos004-vllm-source"
        def function(filename, class_name, name, extras=None):
            tree = ast.parse((folder / filename).read_text())
            owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
            method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)
            namespace = {"Any": object, **(extras or {})}
            exec(compile(ast.Module(body=[method], type_ignores=[]), filename, "exec"), namespace)
            extracted = namespace[name]
            if extras and name in extras:
                namespace[name] = extras[name]
            return extracted
        source_generation = {"eos_token_id": [128001, 128009]}
        called = []
        get_generation = function("vllm__config__model.py", "ModelConfig", "try_get_generation_config",
            {"try_get_generation_config": lambda *a, **kw: called.append(a) or SimpleNamespace(to_diff_dict=lambda: dict(source_generation))})
        fields = SimpleNamespace(model="derived003", hf_config_path=None, generation_config="vllm", trust_remote_code=False, revision=None, config_format="auto", override_generation_config={})
        fields.try_get_generation_config = lambda: get_generation(fields)
        get_defaults = function("vllm__config__model.py", "ModelConfig", "get_diff_sampling_param")
        self.assertEqual(get_defaults(fields), {})
        self.assertEqual(get_generation(fields), source_generation)
        self.assertEqual(called, [("derived003",)])
        update = function("vllm__sampling_params.py", "SamplingParams", "update_from_generation_config")
        for generation, expected in ((source_generation, {128001, 128009}), ({"eos_token_id": 128009}, {128009})):
            params = SimpleNamespace(ignore_eos=False, stop_token_ids=[128009], _all_stop_token_ids={128009})
            update(params, generation, 128009)
            self.assertEqual(params._all_stop_token_ids, expected)
            self.assertEqual(params._eos_token_id, 128009)

    def test_128001_may_continue_but_only_128009_may_stop(self):
        from mobilevla_compare_policy006_004 import stop_boundary_coverage
        policy = {"max_new_tokens": 4, "eos_token_ids": [128009]}
        self.assertEqual(validate_completion([31, 128001, 32, 128009], "stop", None, policy), [])
        self.assertIn("stop_finish_without_terminal_EOS", validate_completion([31, 128001], "stop", 128001, policy))
        coverage = stop_boundary_coverage({"generated_token_ids": [31, 128001, 32, 128009], "finish_reason": "stop", "generation_policy": policy})
        self.assertTrue(coverage["continued_after_nonpublisher128001"])
        self.assertTrue(coverage["publisher128009_terminal_EOS_observed"])
        self.assertFalse(stop_boundary_coverage({"generated_token_ids": [31, 32], "finish_reason": "length"})["stop_boundary_exercised"])


if __name__ == "__main__":
    unittest.main()
