from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.evaluate_xcopa_minilm_zero_shot import _git_blob_sha, _load_rows, _metrics


def test_git_blob_hash_matches_git_object_formula() -> None:
    assert _git_blob_sha(b"test\n") == "9daeafb9864cf43055ae93beb0afd6c7d144bfa4"


def test_load_rows_rejects_nonvalidation_count(tmp_path: Path) -> None:
    path = tmp_path / "val.et.jsonl"
    path.write_text(json.dumps({"idx": 0}), encoding="utf-8")
    with pytest.raises(ValueError, match="expected 100"):
        _load_rows(path, "et")


def test_load_rows_rejects_unexpected_schema(tmp_path: Path) -> None:
    row = {
        "premise": "P",
        "choice1": "A",
        "choice2": "B",
        "question": "effect",
        "label": 0,
        "idx": 0,
        "changed": False,
    }
    path = tmp_path / "val.et.jsonl"
    path.write_text("\n".join(json.dumps(row | {"idx": i}) for i in range(100)), encoding="utf-8")
    assert len(_load_rows(path, "et")) == 100
    path.write_text(
        "\n".join(json.dumps(row | {"idx": i, "extra": 1}) for i in range(100)), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="schema"):
        _load_rows(path, "et")


def test_metrics_match_binary_choice_probabilities() -> None:
    metrics = _metrics([torch.tensor([2.0, 0.0]), torch.tensor([0.0, 2.0])], [0, 0])
    assert metrics["count"] == 2
    assert metrics["accuracy"] == 0.5
    assert metrics["nll"] > 0.5
    assert 0.0 <= metrics["brier"] <= 2.0
    assert 0.0 <= metrics["ece_15_bins"] <= 1.0


def test_metrics_empty_group_uses_null_values() -> None:
    metrics = _metrics([], [])
    assert metrics == {
        "count": 0,
        "accuracy": None,
        "nll": None,
        "brier": None,
        "ece_15_bins": None,
    }
