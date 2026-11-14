from __future__ import annotations

from scripts.audit_clotho_aqa_group_split import build_split


def _rows() -> list[dict[str, object]]:
    rows = []
    for group in range(6):
        for clip in range(2):
            for question in range(2):
                votes = [group % 2, group % 2, (group + question) % 2]
                rows.append(
                    {
                        "file_name": f"clip-{group}-{clip}.wav",
                        "source_id": f"source-{group}-{clip}",
                        "uploader": f"uploader-{group}",
                        "question": f"Question {question}?",
                        "target": int(sum(votes) >= 2),
                        "votes": votes,
                        "license": "http://creativecommons.org/publicdomain/zero/1.0/",
                    }
                )
    return rows


def test_group_split_is_deterministic_and_disjoint() -> None:
    first = build_split(_rows(), seed=17, dev_groups=2, calibration_groups=1)
    second = build_split(_rows(), seed=17, dev_groups=2, calibration_groups=1)

    assert first == second
    summary, members = first
    assert summary["cross_split_uploader_overlap"] == 0
    assert summary["cross_split_audio_filename_overlap"] == 0
    assert summary["cross_split_source_id_overlap"] == 0
    assert {name: item["groups"] for name, item in summary["splits"].items()} == {
        "train": 3,
        "calibration": 1,
        "dev": 2,
    }
    assert len(members) == len(_rows())


def test_question_overlap_is_reported_and_votes_are_preserved_as_counts() -> None:
    summary, members = build_split(_rows(), seed=17, dev_groups=2, calibration_groups=1)

    assert any(summary["cross_split_normalized_question_overlap"].values())
    assert all(0 <= row["yes_votes"] <= row["vote_count"] == 3 for row in members)
    assert all(len(row["group_id"]) == 64 for row in members)


def test_duplicate_audio_question_identity_fails_closed() -> None:
    rows = _rows()
    rows.append(rows[0].copy())

    try:
        build_split(rows, seed=17, dev_groups=2, calibration_groups=1)
    except ValueError as error:
        assert "duplicate audio-question" in str(error)
    else:
        raise AssertionError("duplicate sample was accepted")


def test_unlicensed_row_fails_closed() -> None:
    rows = _rows()
    rows[0]["license"] = "unknown"

    try:
        build_split(rows, seed=17, dev_groups=2, calibration_groups=1)
    except ValueError as error:
        assert "exact clip licenses" in str(error)
    else:
        raise AssertionError("unlicensed row was accepted")


def test_vote_target_mismatch_fails_closed() -> None:
    rows = _rows()
    rows[0]["target"] = 1 - int(rows[0]["target"])

    try:
        build_split(rows, seed=17, dev_groups=2, calibration_groups=1)
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("inconsistent target was accepted")
