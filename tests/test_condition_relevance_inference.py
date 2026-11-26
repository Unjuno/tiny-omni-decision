import numpy as np
import pytest


def test_scoring_preserves_rows_candidates_and_masks_unsupplied_choices():
    from scripts.condition_relevance_inference import score_rows

    rows = [
        dict(state="s1", question="q1", options=["a", "b"], target="a", private="hidden"),
        dict(state="s2", question="q2", options=["c", "d", "e"], target="c"),
    ]
    observed = []

    def tokenizer(first, second, **kwargs):
        assert kwargs == dict(
            padding=True, truncation="only_second", max_length=512, return_tensors="pt"
        )
        observed.extend(zip(first, second, strict=True))
        return {"pairs": list(zip(first, second, strict=True))}

    def forward(encoded):
        return np.asarray([ord(first[-1]) for first, _ in encoded["pairs"]], dtype=np.float32)

    scores = score_rows(rows, tokenizer, forward, pair_batch=3, max_length=512)
    assert np.array_equal(scores[0, :2], [97, 98])
    assert scores[0, 2] <= -1e29
    assert np.array_equal(scores[1], [99, 100, 101])
    assert len(observed) == 5
    assert all("hidden" not in a + b and "target" not in a + b for a, b in observed)


@pytest.mark.parametrize("bad", [np.asarray([float("nan"), 0]), np.asarray([0])])
def test_bad_forward_outputs_rejected(bad):
    from scripts.condition_relevance_inference import score_rows

    rows = [dict(state="s", question="q", options=["a", "b"])]
    with pytest.raises(ValueError):
        score_rows(rows, lambda *a, **k: {}, lambda encoded: bad, pair_batch=32, max_length=512)


def test_byte_identical_weights_share_one_forward_but_keep_all_roles():
    from scripts.condition_relevance_inference import evaluation_groups

    roles = {}
    for name, arm, step, h in [
        ("initial", "ce", 0, "a"),
        ("ce_endpoint", "ce", 1024, "b"),
        ("ce_selected", "ce", 0, "a"),
        ("relevance_endpoint", "relevance", 1024, "c"),
        ("relevance_selected", "relevance", 0, "a"),
    ]:
        roles[name] = dict(arm=arm, step=step, filename=f"step-{step}-delta.pt", sha256=h * 64)
    groups = evaluation_groups(roles)
    assert len(groups) == 3
    assert groups[0]["roles"] == ["initial", "ce_selected", "relevance_selected"]
    assert sum(len(g["roles"]) for g in groups) == 5


def test_missing_role_is_rejected_even_if_its_model_duplicates_another():
    from scripts.condition_relevance_inference import evaluation_groups

    with pytest.raises(ValueError):
        evaluation_groups({"initial": {"sha256": "a" * 64}})
