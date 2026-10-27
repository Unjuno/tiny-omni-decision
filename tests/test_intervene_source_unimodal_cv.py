from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from scripts.intervene_source_unimodal_cv import donor_mapping  # noqa: E402


def test_donors_are_fixed_and_from_different_video():
    rows = [
        SimpleNamespace(video_id="b", record_id="b2"),
        SimpleNamespace(video_id="a", record_id="a2"),
        SimpleNamespace(video_id="a", record_id="a1"),
    ]
    assert donor_mapping(rows) == {"a1": "b2", "a2": "b2", "b2": "a1"}
    assert donor_mapping(rows) == donor_mapping(list(reversed(rows)))


def test_single_video_rejected():
    with pytest.raises(ValueError):
        donor_mapping([SimpleNamespace(video_id="a", record_id="a1")])
