from collections import namedtuple

import pytest

pytest.importorskip("torch")
from scripts import train_source_set_cv  # noqa: E402
from scripts.source_set_metrics import score_record  # noqa: E402
from scripts.summarize_source_set_cv import summarize  # noqa: E402


def test_disk_guard_stops_before_output_creation(monkeypatch, tmp_path):
    monkeypatch.setattr(train_source_set_cv, "ROOT", tmp_path)
    usage = namedtuple("usage", "total used free")(2**30, 0, 2**30)
    monkeypatch.setattr(train_source_set_cv.shutil, "disk_usage", lambda _: usage)
    monkeypatch.setattr("sys.argv", ["runner", "--output", str(tmp_path / "new")])
    with pytest.raises(OSError, match="disk reserve"):
        train_source_set_cv.main()
    assert not (tmp_path / "new").exists()


def test_summary_checks_saved_metrics():
    row = {
        "record_id": "one",
        "video_id": "v",
        "candidate_ids": ["a", "b"],
        "target_candidate_ids": ["a", "b"],
        "selected": ["a"],
        "conditional_probabilities": [0.8, 0.6],
        "none_probability": 0.1,
    }
    row.update(
        score_record(
            row["candidate_ids"],
            row["target_candidate_ids"],
            row["selected"],
            row["conditional_probabilities"],
            row["none_probability"],
        )
    )
    report = summarize([row])
    assert report["exact_set_accuracy"] == 0
    assert report["candidate_binary_micro"]["n"] == 2
    row["exact"] = True
    with pytest.raises(ValueError, match="saved metric"):
        summarize([row])
