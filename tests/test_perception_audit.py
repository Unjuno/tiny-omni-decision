import pytest

from scripts.audit_perception_train import summarize


def test_train_only_and_visibility_counting():
    row = dict(
        metadata=dict(split="train", video_id="v", num_frames=10),
        mc_question=[],
        object_tracking=[dict(id=1)],
        sound_localisation=[
            dict(label="impact", is_visible=0, parent_objects=[1], timestamps=[0, 100])
        ],
    )
    report = summarize({"v": row}, {"v": 5})
    assert report["counts"]["cut_required_videos"] == 1
    assert report["counts"]["videos_with_known_sound_parent"] == 1
    assert report["sound_visibility"]["0"] == 1
    row["metadata"]["split"] = "valid"
    with pytest.raises(ValueError, match="train-only"):
        summarize({"v": row}, {"v": 5})
