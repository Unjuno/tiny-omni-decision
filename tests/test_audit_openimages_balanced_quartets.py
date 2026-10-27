import pytest

pytest.importorskip("torch")
from scripts.audit_openimages_balanced_quartets import build_quartets  # noqa: E402


def test_quartet_has_identical_options_all_targets_and_unique_images():
    scope = set("abcd")
    rows = [{"image_id": label, "positive_labels": [label],
             "negative_labels": sorted(scope - {label})} for label in sorted(scope)]
    groups, sets, full, unused = build_quartets(rows, set(scope), scope)
    assert (len(groups), sets, full, unused) == (1, 1, 1, 0)
    assert set(groups[0]["options"]) == scope
    assert {row["target"] for row in groups[0]["rows"]} == scope
    assert len({row["image_id"] for row in groups[0]["rows"]}) == 4
    assert groups == build_quartets(rows, set(scope), scope)[0]


def test_quartet_fails_closed_on_conflicting_or_missing_data():
    bad = [{"image_id": "a", "positive_labels": ["a"], "negative_labels": ["a"]}]
    with pytest.raises(ValueError, match="verified labels"):
        build_quartets(bad, {"a"}, {"a"})
    with pytest.raises(ValueError, match="identity"):
        build_quartets([], {"a"}, {"a"})
