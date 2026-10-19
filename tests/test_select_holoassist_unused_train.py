from __future__ import annotations

import zipfile

import pytest

from scripts.select_holoassist_unused_train import _read_official_train, make_selection


def _inputs():
    ids = {f"session-{task}-{index}" for task in ("rare", "common") for index in range(5)}
    annotations = [
        {"video_name": session, "taskType": session.split("-")[1]}
        for session in sorted(ids)
    ]
    members = [
        {
            "session_id": session,
            "name": f"{session}/Export_py/Video_pitchshift.mp4",
            "offset_data": index * 100,
            "size": 100,
        }
        for index, session in enumerate(sorted(ids))
    ]
    return ids, annotations, members


def test_selection_is_deterministic_and_partitions_unused_official_train():
    ids, annotations, members = _inputs()
    args = dict(
        official_train=ids,
        annotations=annotations,
        old_excluded={"session-common-0"},
        members=members,
        seed=20261019,
    )
    first = make_selection(**args)
    second = make_selection(**args)
    assert first["selection"] == second["selection"]
    train = {row["session_id"] for row in first["selection"]["train"]["videos"]}
    validation = {
        row["session_id"] for row in first["selection"]["validation"]["videos"]
    }
    assert not train & validation
    assert train | validation == ids - {"session-common-0"}
    assert "session-common-0" not in train | validation


def test_selection_fails_when_an_unused_train_session_has_no_media_member():
    ids, annotations, members = _inputs()
    with pytest.raises(ValueError, match="lacks indexed pitch-shifted media"):
        make_selection(
            official_train=ids,
            annotations=annotations,
            old_excluded=set(),
            members=members[:-1],
            seed=17,
        )


def test_selection_uses_only_allowlisted_pitch_shift_member_paths():
    ids, annotations, members = _inputs()
    members[0]["name"] = f"{members[0]['session_id']}/Export_py/Video_compress.mp4"
    with pytest.raises(ValueError, match="lacks indexed pitch-shifted media"):
        make_selection(
            official_train=ids,
            annotations=annotations,
            old_excluded=set(),
            members=members,
            seed=17,
        )


def test_selector_reads_only_official_train_split_member(tmp_path):
    path = tmp_path / "official-splits.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("train-v1_2.txt", "train-a\n")
        archive.writestr("val-v1_2.txt", b"\xff\xfe")
        archive.writestr("test-v1_2.txt", b"\xff\xfe")

    assert _read_official_train(path) == {"train-a"}
