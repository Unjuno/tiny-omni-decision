import importlib
import importlib.util
import json

import pytest


def _selector():
    spec = importlib.util.find_spec("scripts.build_ami_development_visual_review")
    assert spec is not None, "development-only AMI visual review selector is not implemented"
    module = importlib.import_module("scripts.build_ami_development_visual_review")
    return module.select_development_samples


def _preflight(meetings=None):
    ids = ["ES2003a", "ES2011a", "IS1008a", "TS3004a", "TS3006a"]
    if meetings is not None:
        ids = meetings
    result = {"meetings": {}}
    for meeting_id in ids:
        result["meetings"][meeting_id] = {
            "partition": "development",
            "parent_set": meeting_id[:-1],
            "camera_map": {"A": "Closeup1", "B": "Closeup2", "C": "Closeup3", "D": "Closeup4"},
            "candidate_windows": {
                "A": [{"start_s": 1.0, "end_s": 2.0}],
                "B": [{"start_s": 3.0, "end_s": 4.0}],
                "C": [],
                "D": [{"start_s": 7.0, "end_s": 8.0}],
            },
        }
    return result


def test_selector_returns_only_pinned_development_windows_in_stable_order():
    data = _preflight()
    data["meetings"]["ES2002a"] = {
        "partition": "training",
        "parent_set": "ES2002",
        "camera_map": {"A": "Closeup1"},
        "candidate_windows": {"A": [{"start_s": 0.0, "end_s": 1.0}]},
    }

    samples = _selector()(data)

    assert len(samples) == 15
    assert samples[0] == {
        "sample_id": "ES2003a:A:0000:1.000:2.000",
        "meeting": "ES2003a",
        "parent_set": "ES2003",
        "partition": "development",
        "agent": "A",
        "target_camera": "Closeup1",
        "start_s": 1.0,
        "end_s": 2.0,
    }
    assert samples[1]["sample_id"] == "ES2003a:B:0000:3.000:4.000"
    assert all(sample["partition"] == "development" for sample in samples)


def test_selector_fails_closed_on_sealed_or_unselected_development_meetings():
    data = _preflight()
    data["meetings"]["SC4001a"] = {
        "partition": "sealed",
        "parent_set": "SC4001",
        "camera_map": {},
        "candidate_windows": {},
    }

    with pytest.raises(ValueError, match="non-development partition"):
        _selector()(data)

    data = _preflight()
    data["meetings"]["XX9999a"] = {
        "partition": "development",
        "parent_set": "XX9999",
        "camera_map": {},
        "candidate_windows": {},
    }
    with pytest.raises(ValueError, match="not in the pinned development allowlist"):
        _selector()(data)


def test_selector_rejects_missing_pinned_development_meeting():
    with pytest.raises(ValueError, match="missing pinned development meetings"):
        _selector()(_preflight(["ES2003a", "ES2011a", "IS1008a", "TS3004a"]))


def test_selector_rejects_invalid_window_and_duplicate_sample_id():
    data = _preflight()
    data["meetings"]["ES2003a"]["candidate_windows"]["A"] = [
        {"start_s": 2.0, "end_s": 1.0}
    ]
    with pytest.raises(ValueError, match="invalid candidate window"):
        _selector()(data)

    data = _preflight()
    data["meetings"]["ES2003a"]["candidate_windows"]["A"] = [
        {"start_s": 1.0, "end_s": 2.0},
        {"start_s": 1.0, "end_s": 2.0},
    ]
    with pytest.raises(ValueError, match="duplicate candidate window"):
        _selector()(data)


def test_merge_preflights_accepts_single_meeting_manifest(tmp_path):
    module = importlib.import_module("scripts.build_ami_development_visual_review")
    path = tmp_path / "single-meeting.json"
    path.write_text(
        json.dumps(
            {
                "meeting": "TS3004a",
                "split": "development",
                "parent_set": "TS3004",
                "camera_map": {},
                "candidate_windows": {},
                "videos": [],
            }
        ),
        encoding="utf-8",
    )

    merged = module.merge_preflights([path])

    assert merged["meetings"]["TS3004a"]["partition"] == "development"
    assert merged["meetings"]["TS3004a"]["parent_set"] == "TS3004"


def test_merge_preflights_rejects_duplicate_meeting(tmp_path):
    module = importlib.import_module("scripts.build_ami_development_visual_review")
    path = tmp_path / "duplicate.json"
    path.write_text(
        json.dumps({"meeting": "ES2003a", "split": "development"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate meeting"):
        module.merge_preflights([path, path])
