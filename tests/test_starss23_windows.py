import pytest

from scripts.probe_starss23_windows import parse_annotations, validate_pairs, window_queries


def test_half_open_windows_preserve_overlap_and_merge_speech():
    rows = parse_annotations("49,0,1,0,0,1\n49,1,2,0,0,1\n49,2,1,0,0,1\n50,3,1,0,0,1")
    before = window_queries(rows, 0, 50)
    after = window_queries(rows, 50, 100)
    assert len(before) == len(after) == 12
    assert before[0]["active_annotation_frames"] == 1
    assert before[0]["target"] == before[1]["target"] == 1
    assert before[2]["target"] == 0
    assert after[2]["target"] == 1
    assert sum(q["target"] for q in after) == 1


@pytest.mark.parametrize("text", ["1,13,0,0,0,1", "-1,0,0,0,0,1", "1,0,0,181,0,1", "1,0"])
def test_invalid_annotations_fail_closed(text):
    with pytest.raises(ValueError):
        parse_annotations(text)


def test_probe_rejects_dev_test():
    report = {
        "status": "PASS",
        "record": "doi:10.5281/zenodo.7880637",
        "pairs": [{"clip": "a", "split": "test"}],
    }
    with pytest.raises(ValueError, match="dev-train"):
        validate_pairs(report)


def test_empty_annotation_window_retains_every_question():
    queries = window_queries([], 0, 50)
    assert len(queries) == 12
    assert all(q["target"] == 0 and q["options"] == ["no", "yes"] for q in queries)
