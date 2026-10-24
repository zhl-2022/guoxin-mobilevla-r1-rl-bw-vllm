"""Controller lifecycle tests with simulated services; no browser/GPU/network."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mobilevla_comparison_orchestrator007_003 as subject


class OrchestratorTests(unittest.TestCase):
    def test_full_original_source_and_helper_locks_match_local_files(self):
        revision = ROOT / "revisions/mobilevla-eos004"
        if not revision.exists(): revision = ROOT
        result = subject.check_sources(revision, revision)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["verified_payload_file_counts"], {"frozen": 70, "comparison": 5})

    def test_parent_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                subject.regular_ancestors(Path(tmp) / ".." / "escape")

    def simulated_main(self, cleanup_status="PASS", native_error=False, prepare_missing=False, stop_coverage=True):
        with tempfile.TemporaryDirectory() as tmp:
            private = Path(tmp).resolve()
            frozen, comparison, source, model = [private / n for n in ("frozen", "comparison", "source", "model")]
            for path in (frozen, comparison, source, model):
                if path != model or not prepare_missing:
                    path.mkdir()
            task = frozen / "tasks/mobilevla-r1-rl-bw-vllm"
            runtime = task / "runtime"
            runtime.mkdir(parents=True)
            receipt = {"source": str(source), "derived": str(model), "schema": "mobilevla-derived-hf-view-007-eos004-v1", "adapter_revision": "007-eos004"}
            if not prepare_missing:
                (model / "prepare-receipt.json").write_text(json.dumps(receipt))
            output = private / "output"
            events = []
            class FakeController:
                def __init__(self, output, deadline, environment):
                    self.output, self.steps = output, []
                def execute(self, name, command, cap, allowed=(0,), cleanup=False):
                    events.append(name)
                    self.steps.append({"name": name, "status": "SIMULATED"})
                    if name == "prepare-derived":
                        model.mkdir()
                        (model / "prepare-receipt.json").write_text(json.dumps(receipt))
                    if name == "native-start":
                        run_id = command[-1]
                        (runtime / "state.json").write_text(json.dumps({"run_id": run_id, "status": "ready"}))
                        (runtime / ("config-" + run_id + ".json")).write_text(json.dumps({"run_id": run_id}))
                    if native_error and name == "native-1":
                        raise RuntimeError("simulated native failure")
                    if name.startswith("compare-"):
                        target = Path(command[command.index("--output") + 1])
                        target.write_text(json.dumps({"comparable": True, "raw_generated_token_ids_equal": True,
                            "native_stop_boundary_coverage": {"stop_boundary_exercised": stop_coverage},
                            "reference_stop_boundary_coverage": {"stop_boundary_exercised": stop_coverage}}))
                    return 0
            def stopped(controller, frozen, run_id, label):
                events.append(label)
                return {"status": cleanup_status}
            argv = ["--output", str(output), "--frozen", str(frozen), "--comparison", str(comparison),
                    "--source", str(source), "--model", str(model)]
            with mock.patch.object(subject.sys, "platform", "linux"), \
                 mock.patch.object(subject, "PRIVATE_ROOT", private), \
                 mock.patch.object(subject, "check_sources", return_value={"status": "PASS"}), \
                 mock.patch.object(subject, "Controller", FakeController), \
                 mock.patch.object(subject, "stop_owned", side_effect=stopped), \
                 mock.patch.object(subject.shutil, "which", return_value=None), \
                 contextlib.redirect_stdout(io.StringIO()):
                code = subject.main(argv)
            return code, json.loads((output / "orchestrator-result.json").read_text()), events

    def test_native_then_verified_cleanup_then_HF_and_both_comparisons(self):
        code, result, events = self.simulated_main()
        self.assertEqual(code, 0)
        self.assertEqual(events, ["native-tokenizer-probe", "native-start", "native-1", "native-8", "native-eos-1", "native-stop", "hf-1", "hf-8", "hf-eos-1", "compare-1", "compare-8", "compare-eos-1", "final-stop"])
        self.assertEqual(len(result["run_id"]), 32)
        self.assertFalse(result["official_accuracy_verified"])

    def test_failed_owned_cleanup_blocks_HF(self):
        code, result, events = self.simulated_main(cleanup_status="FAIL")
        self.assertEqual(code, 2)
        self.assertNotIn("hf-1", events)
        self.assertEqual(events[-1], "final-stop")
        self.assertEqual(result["status"], "failed_owned_cleanup")

    def test_equal_32_token_prefixes_cannot_replace_long_stop_boundary_coverage(self):
        code, result, events = self.simulated_main(stop_coverage=False)
        self.assertEqual(code, 2)
        self.assertEqual(result["status"], "completed_equal_without_stop_boundary_coverage")
        self.assertFalse(result["publisher_stop_boundary_exercised"])
        self.assertEqual(events[-1], "final-stop")

    def test_native_failure_still_stops_and_never_starts_HF(self):
        code, result, events = self.simulated_main(native_error=True)
        self.assertEqual(code, 2)
        self.assertEqual(events, ["native-tokenizer-probe", "native-start", "native-1", "native-stop", "final-stop"])
        self.assertIn("simulated native failure", result["error"])

    def test_missing_003_derived_view_is_prepared_before_native_launch(self):
        code, result, events = self.simulated_main(prepare_missing=True)
        self.assertEqual(code, 0)
        self.assertEqual(events[:3], ["prepare-derived", "native-tokenizer-probe", "native-start"])
        self.assertEqual(events[-1], "final-stop")

    def test_stop_refuses_to_signal_a_different_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            frozen = Path(tmp) / "frozen"
            state = frozen / "tasks/mobilevla-r1-rl-bw-vllm/runtime/state.json"
            state.parent.mkdir(parents=True)
            state.write_text(json.dumps({"run_id": "a" * 32, "status": "ready"}))
            output = Path(tmp) / "output"
            output.mkdir()
            controller = mock.Mock(output=output)
            result = subject.stop_owned(controller, frozen, "b" * 32, "stop")
            self.assertEqual(result["status"], "UNKNOWN")
            controller.execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
