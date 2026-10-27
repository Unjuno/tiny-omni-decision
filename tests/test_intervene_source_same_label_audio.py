import pytest

pytest.importorskip("torch")

from scripts.intervene_source_same_label_audio import same_label_donors  # noqa: E402


def test_donors_keep_label_and_fold_and_skip_singletons():
    rows = [
        {"record_id": "a1", "video_id": "a", "label": 4, "fold": 0},
        {"record_id": "b1", "video_id": "b", "label": 4, "fold": 0},
        {"record_id": "c1", "video_id": "c", "label": 4, "fold": 1},
        {"record_id": "d1", "video_id": "d", "label": 5, "fold": 0},
        {"record_id": "e1", "video_id": "e", "label": None, "fold": 0},
    ]
    assert same_label_donors(rows) == {"a1": "b1", "b1": "a1"}
    assert same_label_donors(rows) == same_label_donors(list(reversed(rows)))
