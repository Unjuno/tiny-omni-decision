import importlib

import pytest


def test_training_video_is_mapped_to_fit_or_selection_without_splitting_groups():
    finalizer = importlib.import_module(
        "scripts.finalize_perception_av_event_feature_manifest"
    )
    train_ids = {"video_1", "video_2"}
    selection_ids = {"video_1"}
    assert finalizer.assign_model_split("train", "video_1", train_ids, selection_ids) == "selection"
    assert finalizer.assign_model_split("train", "video_2", train_ids, selection_ids) == "fit"
    assert (
        finalizer.assign_model_split("calibration", "video_3", train_ids, selection_ids)
        == "calibration"
    )
    assert finalizer.assign_model_split(
        "development_validation", "video_4", train_ids, selection_ids
    ) == "development_validation"


def test_unplanned_training_video_fails_closed():
    finalizer = importlib.import_module(
        "scripts.finalize_perception_av_event_feature_manifest"
    )
    with pytest.raises(ValueError, match="not in the pinned training video set"):
        finalizer.assign_model_split("train", "video_unknown", {"video_1"}, {"video_1"})
