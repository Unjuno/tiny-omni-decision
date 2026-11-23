from __future__ import annotations

import hashlib

import numpy as np
import pytest

from scripts.pretrained_reuse.analyze_socialiqa_minilm_probe import (
    grouped_bootstrap,
    score,
    verify_predictions,
)


def _item(identifier: str, context: str, target: int) -> dict[str, object]:
    normalized = " ".join(context.casefold().split())
    return {"id": identifier, "group": normalized, "target": target}


def _row(item: dict[str, object], probabilities: list[float]) -> dict[str, object]:
    return {
        "id": item["id"],
        "target": item["target"],
        "context_group_sha256": hashlib.sha256(str(item["group"]).encode()).hexdigest(),
        "prediction": int(np.argmax(probabilities)),
        "probabilities": probabilities,
    }


def test_prediction_integrity_checks_ids_group_and_probabilities() -> None:
    items = [_item("dev-0", "same context", 1)]
    probs = np.asarray(verify_predictions([_row(items[0], [0.1, 0.8, 0.1])], items))
    np.testing.assert_allclose(probs, [[0.1, 0.8, 0.1]])
    broken = _row(items[0], [0.2, 0.2, 0.2])
    with pytest.raises(ValueError, match="sum to one"):
        verify_predictions([broken], items)


def test_score_and_grouped_bootstrap_are_deterministic() -> None:
    items = [
        _item("0", "group a", 0),
        _item("1", "group a", 1),
        _item("2", "group b", 2),
    ]
    first = np.asarray([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.2, 0.2, 0.6]])
    second = np.asarray([[0.4, 0.3, 0.3], [0.1, 0.7, 0.2], [0.6, 0.2, 0.2]])
    labels = np.asarray([0, 1, 2])
    assert score(first, labels)["accuracy"] == 1.0
    a = grouped_bootstrap(items, first, second, replicates=100, seed=1729)
    b = grouped_bootstrap(items, first, second, replicates=100, seed=1729)
    assert a == b
    assert a["groups"] == 2
