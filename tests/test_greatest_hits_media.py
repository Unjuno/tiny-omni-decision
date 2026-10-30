import pytest


def test_media_selection_excludes_quarantine_and_labels():
    from scripts.greatest_hits_media import planned_media

    plan = {"train": ["a"], "validation": ["b"], "quarantine": ["c"]}
    entries = {"vis-data-256/" + i + "_denoised.mp4": dict(
        name="vis-data-256/" + i + "_denoised.mp4", bytes=100, compressed=80) for i in "abc"}
    result = planned_media(plan, entries)
    assert [r["parent"] for r in result] == ["a", "b"]
    assert [r["split"] for r in result] == ["train", "validation"]
    with pytest.raises(ValueError):
        planned_media({**plan, "validation": ["a"]}, entries)
    entries["vis-data-256/b_denoised.mp4"]["bytes"] = 40 * 2**20
    with pytest.raises(ValueError):
        planned_media(plan, entries)


def test_window_accounting_retains_out_of_bounds_rows():
    from scripts.greatest_hits_media import window_integrity

    rows = [dict(seconds=t, fields=["wood", "hit", None]) for t in [0.1, 1.0, 2.9, 3.1]]
    result = window_integrity(rows, (0.0, 3.0), (0.0, 2.95))
    assert result["rows"] == 4 and result["valid_windows"] == 1
    assert result["out_of_bounds_onsets"] == [3]
    assert result["out_of_bounds_windows"] == [0, 2, 3]
    assert len(rows) == 4


def test_decoded_identity_overlap_fails_across_development_splits():
    from scripts.greatest_hits_media import reject_cross_split_content

    rows = [dict(parent=p, split=s, video_sha256=v, audio_sha256=a)
            for p, s, v, a in [("a", "train", "v1", "a1"),
                               ("b", "validation", "v2", "a2")]]
    reject_cross_split_content(rows)
    rows[1]["audio_sha256"] = "a1"
    with pytest.raises(ValueError, match="audio_sha256"):
        reject_cross_split_content(rows)
