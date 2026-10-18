"""Fail-closed runner tests with temporary metadata and no model/GPU import."""

import builtins
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import evaluate_semantic_native_comparison as runner


class PrerequisiteTest(unittest.TestCase):
    def test_source_inventory_has_bounded_command_length_and_preserves_every_path(self):
        names = [f"scripts/nested_{i:04d}/" + "long_path_" * 8 + ".py" for i in range(1000)]
        names += [
            "scripts/evaluate_semantic_native_comparison.py",
            "scripts/semantic_native_comparison.py",
            "configs/config.json", "pyproject.toml", "docs/protocol.md",
        ]
        with patch.object(
            runner.subprocess, "check_output", return_value="\0".join(names) + "\0"
        ) as call:
            actual = runner.tracked_source_names(
                Path("repo"), "configs/config.json", "docs/protocol.md"
            )
        self.assertEqual(set(actual), set(names))
        self.assertLess(sum(len(str(x)) for x in call.call_args.args[0]), 512)
        self.assertIn("-z", call.call_args.args[0])

    def test_source_inventory_rejects_untracked_required_file(self):
        names = [
            "scripts/evaluate_semantic_native_comparison.py",
            "scripts/semantic_native_comparison.py",
            "configs/config.json", "pyproject.toml", "docs/protocol.md",
        ]
        for missing in names:
            with patch.object(
                runner.subprocess, "check_output",
                return_value="\0".join(n for n in names if n != missing) + "\0",
            ):
                with self.assertRaisesRegex(ValueError, "tracked and frozen"):
                    runner.tracked_source_names(
                        Path("repo"), "configs/config.json", "docs/protocol.md"
                    )

    def test_native_payload_coverage_required(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "unpinned native input"):
                runner.admit_native_sources(Path(folder), {})

    def attempt(self, drop_pin=False):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            primary = root / "decision-semantic-fit-01/result.json"
            replay = root / "decision-semantic-selected-replay-01/result.json"
            independent = root / "decision-semantic-independent-verification-01.json"
            for p, value in (
                (primary, {"status": "RUNNING", "steps": 768}),
                (replay, {"status": "PASS_SELECTED_SEMANTIC_REPLAY"}),
                (independent, {"status": "PASS_INDEPENDENT_SEMANTIC_VERIFICATION"}),
            ):
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(value), encoding="utf8")
            pins = {str(p): runner.sha(p) for p in (primary, replay, independent)}
            if drop_pin:
                del pins[str(independent)]
            config = root / "config.json"
            config.write_text(
                json.dumps(dict(input_pins=pins, artifact_root=str(root))), encoding="utf8"
            )
            output = root / "output"
            original_import = builtins.__import__

            def reject_gpu_import(name, *args, **kwargs):
                if name == "torch" or name.startswith("torch."):
                    raise AssertionError("must reject before torch import")
                return original_import(name, *args, **kwargs)

            with (
                patch.object(runner, "CONFIG", config),
                patch.object(sys, "argv", ["native-comparison", "--output", str(output)]),
                patch.object(builtins, "__import__", reject_gpu_import),
            ):
                with self.assertRaisesRegex(
                    ValueError, "not pinned" if drop_pin else "completed fit"
                ):
                    runner.main()
            result = json.loads((output / "result.json").read_text())
            self.assertEqual(result["status"], "INCOMPLETE")
            self.assertEqual(result["optimizer_updates"], 0)
            self.assertFalse(result["transfer_loaded"])
            self.assertFalse(result["sealed_audit_loaded"])

    def test_live_primary_rejected_before_gpu_import(self):
        self.attempt()

    def test_unpinned_terminal_prerequisite_rejected(self):
        self.attempt(drop_pin=True)


if __name__ == "__main__":
    unittest.main()
