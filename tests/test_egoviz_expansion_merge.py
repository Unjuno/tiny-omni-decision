import numpy as np
import pytest

from scripts.merge_egoviz_train_expansion import validate_references, validate_separation


def test_new_training_groups_cannot_overlap_existing_development_or_training():
    current = [
        {"split": "train", "group_id": "a", "chunk_id": "x", "sample_id": "1"},
        {"split": "development", "group_id": "b", "chunk_id": "y", "sample_id": "2"},
    ]
    extra = [{"split": "train", "group_id": "c", "chunk_id": "z", "sample_id": "3"}]
    validate_separation(current, extra)
    for field in ("group_id", "chunk_id", "sample_id"):
        contaminated = [dict(extra[0], **{field: current[1][field]})]
        with pytest.raises(ValueError, match=field):
            validate_separation(current, contaminated)
    with pytest.raises(ValueError, match="training only"):
        validate_separation(current, [dict(extra[0], split="development")])


def test_current_train_development_overlap_is_rejected():
    current = [
        {"split": "train", "group_id": "a", "chunk_id": "x", "sample_id": "1"},
        {"split": "development", "group_id": "a", "chunk_id": "y", "sample_id": "2"},
    ]
    with pytest.raises(ValueError, match="group_id"):
        validate_separation(
            current, [{"split": "train", "group_id": "c", "chunk_id": "z", "sample_id": "3"}]
        )


def test_candidate_target_must_match_query_inside_its_source_session():
    sessions = [{"chunk_id": "x", "group_id": "a", "split": "train", "window_count": 10}]
    row = {
        "chunk_id": "x",
        "group_id": "a",
        "split": "train",
        "query_index": 0,
        "candidate_window_indices": [0, 3, 6, 9],
        "target_option_index": 0,
    }
    validate_references([row], sessions)
    with pytest.raises(ValueError, match="candidate/target"):
        validate_references([dict(row, target_option_index=1)], sessions)
    with pytest.raises(ValueError, match="candidate/target"):
        validate_references([dict(row, candidate_window_indices=[0, 1, 6, 9])], sessions)


def test_development_feature_parity_rejects_changes():
    from scripts.merge_egoviz_train_expansion import verify_development_features

    rows = [{"sample_id": "dev", "candidate_window_indices": [0, 1]}]
    old_video = {"sample_id": np.array(["dev"]), "features": np.array([[1.0, 2.0]])}
    new_video = {
        "sample_id": np.array(["other", "dev"]),
        "features": np.array([[0.0, 0.0], [1.0, 2.0]]),
    }
    old_sensor = np.array([[3.0], [4.0]])
    new_sensor = np.array([[3.0], [4.0], [5.0]])
    verify_development_features(rows, old_video, old_sensor, new_video, new_sensor)
    new_video["features"][1, 0] = 9
    with pytest.raises(ValueError, match="video"):
        verify_development_features(rows, old_video, old_sensor, new_video, new_sensor)
    new_video["features"][1, 0] = 1
    new_sensor[1, 0] = 9
    with pytest.raises(ValueError, match="sensor"):
        verify_development_features(rows, old_video, old_sensor, new_video, new_sensor)
