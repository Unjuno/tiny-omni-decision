import numpy as np
import pytest

from scripts.pretrained_reuse.evaluate_haptile_kinematic_persistence_train import (
    load_train_cache,
    predict_constant_velocity,
    summarize_train_predictions,
)
from tiny_omni_decision.haptile_action import ACTION_LABELS


def test_constant_velocity_projects_signed_axis_and_hold_deadband():
    timestamps = np.arange(8, dtype=np.float64) * (1000.0 / 7.0)
    state = np.zeros((8, 25), dtype=np.float64)
    state[-2, 0] = 0.01
    state[-1, 0] = 0.02
    assert predict_constant_velocity(state, timestamps) == "+x"
    state[-1, 0] = state[-2, 0] + 0.0001
    assert predict_constant_velocity(state, timestamps) == "hold"


@pytest.mark.parametrize(
    ("values", "timestamps"),
    [
        (np.zeros((7, 25)), np.arange(8)),
        (np.zeros((8, 25)), np.array([0, 1, 2, 3, 4, 5, 7, 6])),
        (np.full((8, 25), np.nan), np.arange(8)),
    ],
)
def test_constant_velocity_rejects_bad_inputs(values, timestamps):
    with pytest.raises(ValueError):
        predict_constant_velocity(values, timestamps)


def test_summary_reports_example_and_macro_task_accuracy_separately():
    result = summarize_train_predictions(
        ["a", "b", "c"],
        ["task-a", "task-a", "task-b"],
        np.array([0, 1, ACTION_LABELS.index("hold")]),
        ["+x", "+x", "hold"],
    )
    assert result["persistence_correct"] == 2
    assert result["persistence_accuracy"] == pytest.approx(2 / 3)
    assert result["persistence_macro_task_accuracy"] == pytest.approx(0.75)
    assert result["tasks_persistence_beats_always_hold"] == 1


def test_summary_rejects_duplicate_sample_ids():
    with pytest.raises(ValueError, match="identities"):
        summarize_train_predictions(["x", "x"], ["t", "t"], np.array([0, 1]), ["-x", "+x"])


def test_cache_loader_rejects_non_train_split_before_reading_metadata(tmp_path):
    candidate = tmp_path / "validation-joined-v1" / "decision-features.npz"
    candidate.parent.mkdir()
    with pytest.raises(ValueError, match="only the frozen train"):
        load_train_cache(candidate)
