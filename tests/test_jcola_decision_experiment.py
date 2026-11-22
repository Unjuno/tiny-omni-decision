from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_jcola_bert_decision import (  # noqa: E402
    CandidateScorer,
    assert_disjoint_groups_and_content,
    normalize_sentence,
    split_by_source,
)


def _rows() -> list[dict[str, object]]:
    return [
        {
            "uid": index,
            "source": f"source-{index // 4}",
            "label": index % 2,
            "sentence": f"文 {index}",
        }
        for index in range(16)
    ]


def test_source_group_split_is_deterministic_and_keeps_each_source_whole() -> None:
    rows = _rows()
    train, validation, train_sources, validation_sources = split_by_source(rows, seed=17)
    repeated = split_by_source(rows, seed=17)

    assert [row["uid"] for row in train] == [row["uid"] for row in repeated[0]]
    assert [row["uid"] for row in validation] == [row["uid"] for row in repeated[1]]
    assert set(train_sources).isdisjoint(validation_sources)
    assert {row["source"] for row in train} == set(train_sources)
    assert {row["source"] for row in validation} == set(validation_sources)
    assert sorted(row["uid"] for row in train + validation) == list(range(16))


def test_cross_partition_source_and_normalized_content_overlap_fail_closed() -> None:
    assert_disjoint_groups_and_content(_rows()[:8], _rows()[8:], "fixture")
    with pytest.raises(ValueError, match="source groups"):
        assert_disjoint_groups_and_content(_rows()[:8], _rows()[4:12], "fixture")

    left = [{"uid": 1, "source": "a", "sentence": "Ａ B"}]
    right = [{"uid": 2, "source": "b", "sentence": "a b"}]
    assert normalize_sentence(left[0]["sentence"]) == normalize_sentence(right[0]["sentence"])
    with pytest.raises(ValueError, match="normalized sentence"):
        assert_disjoint_groups_and_content(left, right, "fixture")


def test_shared_option_scorer_permutation_only_permutes_scores() -> None:
    torch.manual_seed(17)
    scorer = CandidateScorer(hidden_size=12, readout_size=5).eval()
    features = torch.randn(7, 2, 12)
    scores = scorer(features)
    swapped = scorer(features[:, [1, 0]])
    assert torch.equal(swapped, scores[:, [1, 0]])
