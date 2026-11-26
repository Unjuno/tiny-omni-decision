import json

import pytest

from tiny_omni_decision.perception_visibility import visibility_records


def row():
    return dict(
        metadata=dict(
            split="train",
            video_id="v",
            frame_rate=10.0,
            audio_sample_rate=100.0,
            audio_samples=1000,
            num_frames=100,
        ),
        sound_localisation=[
            dict(id=0, label="impact", timestamps=[0, 2000000], is_visible=1, parent_objects=[9])
        ],
    )


def test_request_has_no_supervision_and_cut_is_conservative():
    data = row()
    record = list(visibility_records("v", data, 10))[0]
    assert record["status"] == "held"
    assert set(record["request"]) == {"task", "video_id", "interval_us", "sound_class", "options"}
    assert "parent_objects" not in json.dumps(record)
    assert list(visibility_records("v", data, -1))[0]["status"] == "candidate"


def test_same_class_overlap_is_held_and_touching_is_not():
    data = row()
    data["sound_localisation"].append(
        dict(id=1, label="impact", timestamps=[1000000, 3000000], is_visible=0)
    )
    assert all(r["status"] == "held" for r in visibility_records("v", data, -1))
    data["sound_localisation"][1]["timestamps"][0] = 2000000
    assert all(r["status"] == "candidate" for r in visibility_records("v", data, -1))
    data["metadata"]["split"] = "valid"
    with pytest.raises(ValueError):
        list(visibility_records("v", data, -1))
