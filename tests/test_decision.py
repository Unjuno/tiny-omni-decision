from __future__ import annotations

import pytest

from tiny_omni_decision.decision_math import (
    brier_score,
    label_token_ids_from_prompt,
    normalize_scores,
    option_label_ids,
    reorder_target,
)
from tiny_omni_decision.schema import BaseModelManifest, TextDecision


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
