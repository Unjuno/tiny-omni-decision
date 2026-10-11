from unittest.mock import Mock

from scripts.freeze_starss23_splits import split_records, train_coverage


def records():
    return [
        {"split": "train", "clip": f"{room}-{i}", "room": room, "member": f"{room}/{i}"}
        for room in ["tau:room12", "tau:room13", "a", "b", "c", "d", "e", "f"]
        for i in range(2)
    ]


def test_split_is_order_independent_room_grouped_and_exposed_safe():
    rows = records()
    left = split_records(rows, set())
    assert left == split_records(list(reversed(rows)), set())
    for room in {r["room"] for r in left}:
        assert len({r["split"] for r in left if r["room"] == room}) == 1
    assert all(r["split"] == "train" for r in left if r["room"] in {"tau:room12", "tau:room13"})
    assert {r["split"] for r in left} == {"train", "calibration", "validation"}


def test_missing_video_and_official_test_are_excluded():
    rows = records() + [{"split": "test", "clip": "heldout", "room": "other"}]
    result = split_records(rows, {"a-0"})
    assert not {"heldout", "a-0"} & {r["clip"] for r in result}


def test_coverage_does_not_read_calibration_or_validation():
    archive = Mock()
    archive.read.return_value = b"0,0,0,0,0,1\n0,1,1,0,0,1\n"
    result = train_coverage(
        archive,
        [
            {"split": "train", "member": "allowed"},
            {"split": "validation", "member": "forbidden"},
            {"split": "calibration", "member": "also-forbidden"},
        ],
    )
    archive.read.assert_called_once_with("allowed")
    assert result[0]["active_ticks"] == 1
    assert result[0]["recordings"] == 1


def test_frozen_split_bytes_match_recorded_hashes():
    import hashlib
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "docs/pretrained_reuse"
    result = json.loads((root / "R2_STARSS23_SPLITS_20261011.json").read_text())
    for split, recorded in result["splits"].items():
        payload = (root / "starss23-splits-v1" / f"{split}.jsonl").read_bytes()
        assert hashlib.sha256(payload).hexdigest() == recorded["sha256"]
