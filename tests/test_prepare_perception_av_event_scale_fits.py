import pytest

from scripts import prepare_perception_av_event_scale_fits as prepare


def _row(name, split, digest):
    return {"id": name, "video_id": f"video_{name}", "split": split, "media_sha256": digest}


def test_comparison_manifests_use_only_new_holdouts_and_add_unique_fit_videos(monkeypatch):
    monkeypatch.setattr(prepare, "OLD_FIT_EVENTS", 2)
    monkeypatch.setattr(prepare, "OLD_FIT_VIDEOS", 2)
    monkeypatch.setattr(
        prepare,
        "NEW_SPLIT_EVENT_COUNTS",
        {"fit": 2, "selection": 1, "calibration": 1, "development_validation": 1},
    )
    old = [_row("old-a", "fit", "1" * 64), _row("old-b", "fit", "2" * 64)]
    new = [
        _row("new-fit-a", "fit", "3" * 64),
        _row("new-fit-b", "fit", "4" * 64),
        _row("select", "selection", "5" * 64),
        _row("cal", "calibration", "6" * 64),
        _row("dev", "development_validation", "7" * 64),
    ]
    result = prepare.build_comparison_rows(old, new, {"fixed_holdout"})
    assert [row["split"] for row in result["old_only"]].count("fit") == 2
    assert [row["split"] for row in result["expanded"]].count("fit") == 4
    assert {row["video_id"] for row in result["expanded"]}.isdisjoint({"fixed_holdout"})


def test_comparison_manifest_rejects_media_overlap(monkeypatch):
    monkeypatch.setattr(prepare, "OLD_FIT_EVENTS", 1)
    monkeypatch.setattr(prepare, "OLD_FIT_VIDEOS", 1)
    monkeypatch.setattr(
        prepare,
        "NEW_SPLIT_EVENT_COUNTS",
        {"fit": 1, "selection": 1, "calibration": 1, "development_validation": 1},
    )
    old = [_row("old", "fit", "a" * 64)]
    new = [
        _row("new", "fit", "a" * 64),
        _row("select", "selection", "b" * 64),
        _row("cal", "calibration", "c" * 64),
        _row("dev", "development_validation", "d" * 64),
    ]
    with pytest.raises(ValueError, match="overlapping video/media"):
        prepare.build_comparison_rows(old, new, set())
