import pytest

from scripts.build_conditional_curriculum import build
from scripts.condition_binding_corpus import all_states, questions_for
from scripts.publish_semantic_teacher_probe import paired_reversal, verify_replay_cohort
from scripts.semantic_teacher_probe import build_probe


def test_reversal_bootstrap_uses_states_and_same_questions():
    binding = [r for s in all_states()[::19][:8] for r in questions_for(s, "train")]
    rows = build_probe(binding, build(17, 8, "train"), 4)
    predictions = [
        dict(
            prediction_id=r["target_id"] if r["condition"] in ("exclusion", "original") else "wrong"
        )
        for r in rows
    ]
    for source in ("binding", "numeric"):
        result = paired_reversal(rows, predictions, source)
        assert result["groups"] == 4
        assert result["point"] == -1.0
        assert result["percentile95"] == [-1.0, -1.0]
        assert result["draws"] == 10000
        assert result["blind_evaluation"] is False


def test_replay_rejects_empty_duplicate_incomplete_and_wrong_order():
    binding = [r for s in all_states()[::19][:8] for r in questions_for(s, "train")]
    rows = build_probe(binding, build(17, 8, "train"), 4)
    first = {
        source: next(r["group"] for r in rows if r["source"] == source)
        for source in ("binding", "numeric")
    }
    replay = [dict(row_id=r["id"]) for r in rows if r["group"] == first[r["source"]]]
    verify_replay_cohort(rows, replay)
    for bad in ([], replay[:-1], replay[::-1], replay[:-1] + replay[:1]):
        with pytest.raises(ValueError):
            verify_replay_cohort(rows, bad)
