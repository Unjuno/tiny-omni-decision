from __future__ import annotations

import math

import pytest

from tiny_omni_decision.decision_math import (
    brier_score,
    expected_calibration_error,
    label_token_ids_from_prompt,
    negative_log_likelihood,
    normalize_scores,
    option_label_ids,
    reorder_target,
)
from tiny_omni_decision.schema import BaseModelManifest, TextDecision


def test_decision_module_import_does_not_require_optional_torch(monkeypatch) -> None:
    import builtins
    import importlib.util
    import sys
    from pathlib import Path

    module_name = "_decision_without_optional_torch"
    module_path = Path(__file__).parents[1] / "src" / "tiny_omni_decision" / "decision.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    original_import = builtins.__import__

    def without_torch(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("importing decision helpers must not require optional torch")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", without_torch)
    spec.loader.exec_module(module)
    assert callable(module.decision_loss)


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text == "X":
            return [1]
        if text in {"XA", "XB"}:
            return [1, 10 if text[-1] == "A" else 11]
        return {"A": [10], "B": [11], "C": [12], "D": [13], "AB": [2, 3]}[text]


def test_text_decision_schema_checks_target_and_options() -> None:
    TextDecision(state="s", question="q", options=["x", "y"], target="x")
    with pytest.raises(ValueError):
        TextDecision(state="s", question="q", options=["x", "y"], target="z")


def test_option_label_ids_require_single_distinct_tokens() -> None:
    assert option_label_ids(FakeTokenizer(), ["A", "B", "C"]) == [10, 11, 12]
    with pytest.raises(ValueError, match="exactly one token"):
        option_label_ids(FakeTokenizer(), ["AB"])


def test_contextual_label_ids_are_a_single_continuation_token() -> None:
    tokenizer = FakeTokenizer()
    ids, prompt_length = label_token_ids_from_prompt(tokenizer, "X", ["A", "B"])
    assert ids == [10, 11]
    assert prompt_length == 1


def test_option_logits_and_probability_normalization() -> None:
    probs = normalize_scores([10.0, 11.0, 12.0])
    assert sum(probs) == pytest.approx(1.0)
    assert probs == pytest.approx([0.09003057, 0.24472847, 0.66524096])


def test_brier_loss() -> None:
    loss = brier_score([[0.8, 0.2], [0.25, 0.75]], [0, 1])
    assert loss == pytest.approx((0.2**2 + 0.2**2 + 0.25**2 + 0.25**2) / 2)


def test_nll_uses_only_the_supplied_options() -> None:
    assert negative_log_likelihood([[0.8, 0.2], [0.25, 0.75]], [0, 1]) == pytest.approx(
        -(math.log(0.8) + math.log(0.75)) / 2
    )


def test_ece_equal_width_top_label_bins_hand_check() -> None:
    # Bin 0.5 contains one correct 0.6 prediction; bin 0.9 contains a wrong 0.9 prediction.
    assert expected_calibration_error([[0.6, 0.4], [0.1, 0.9]], [0, 0], n_bins=5) == pytest.approx(
        (0.4 + 0.9) / 2
    )


def test_ece_places_confidence_one_in_last_bin() -> None:
    assert expected_calibration_error([[1.0, 0.0]], [0], n_bins=10) == pytest.approx(0.0)


def test_reorder_target_tracks_option_mapping() -> None:
    reordered, target = reorder_target(["a", "b", "c"], "b", [2, 1, 0])
    assert reordered == ["c", "b", "a"]
    assert reordered[target] == "b"


def test_pinned_base_model_manifest_is_valid() -> None:
    manifest = BaseModelManifest.model_validate(
        {
            "schema_version": 1,
            "repo_id": "google/gemma-4-E2B-it-qat-q4_0-unquantized",
            "revision": "6befbaca7398925921802abd1f277b495b78b738",
            "role": "backbone",
            "license": "Apache-2.0",
            "modalities": ["text"],
            "processor_repo_id": "google/gemma-4-E2B-it-qat-q4_0-unquantized",
            "processor_revision": "6befbaca7398925921802abd1f277b495b78b738",
        }
    )
    assert manifest.revision == manifest.processor_revision


def test_model_manifest_requires_matching_immutable_revisions() -> None:
    data = {
        "repo_id": "google/model",
        "revision": "6befbaca7398925921802abd1f277b495b78b738",
        "processor_revision": "main",
        "role": "backbone",
        "license": "Apache-2.0",
        "modalities": ["text"],
    }
    with pytest.raises(ValueError):
        BaseModelManifest.model_validate(data)
