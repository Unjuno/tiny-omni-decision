import pytest

from scripts.publish_teacher_label_factorial import verify_cohort, verify_endpoints


def test_exact_predeclared_replay_cohort_required():
    rows = [
        dict(id=f"{s}:{g}:{i}", source=s, group=str(g))
        for s, n in (("binding", 16), ("numeric", 32))
        for g in range(2)
        for i in range(n)
    ]
    replay = [dict(row_id=r["id"]) for r in rows if r["group"] == "0"]
    verify_cohort(rows, replay)
    for bad in (replay[:-1], replay[::-1], replay[:-1] + replay[:1]):
        with pytest.raises(ValueError):
            verify_cohort(rows, bad)


def test_historical_endpoint_checks_every_record_and_semantic_identity():
    row = dict(id="new", baseline_row_id="old", target_id="a", option_ids=["a", "b"])
    item = dict(
        row_id="new",
        baseline_row_id="old",
        logits=[1.0, 2.0],
        prediction_id="b",
        target_id="a",
    )
    old = {"old": dict(item, row_id="old")}
    assert verify_endpoints([row], [item], old, expected_count=1) == 0
    for changed in (
        dict(item, logits=[1.0, 2.1]),
        dict(item, prediction_id="a"),
        dict(item, baseline_row_id="different"),
        dict(item, target_id="b"),
    ):
        with pytest.raises(ValueError):
            verify_endpoints([row], [changed], old, expected_count=1)
    with pytest.raises(ValueError):
        verify_endpoints([row], [item], old, expected_count=2)
