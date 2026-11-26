from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("torch")

from scripts.fit_perception_av_temporal_scale import (
    _data_for_arm,
    _set_effective_config_status,
    make_training_cohorts,
    paired_metric_interval,
)


def _row(sample_id: str, video_id: str) -> dict[str, object]:
    return {"sample_id": sample_id, "video_id": video_id, "target": 0}


def test_make_training_cohorts_requires_nested_small_set_and_disjoint_validation():
    train = [_row("s1", "v1"), _row("s2", "v2"), _row("s3", "v3")]
    validation = [_row("s4", "v4")]
    manifest = {"train_small": [{"sample_id": "s1"}, {"sample_id": "s3"}]}

    small, large = make_training_cohorts(manifest, train, validation)

    assert [row["sample_id"] for row in small] == ["s1", "s3"]
    assert [row["sample_id"] for row in large] == ["s1", "s2", "s3"]


@pytest.mark.parametrize(
    ("manifest", "train", "validation", "message"),
    [
        ({"train_small": [{"sample_id": "unknown"}]}, [_row("s1", "v1")], [], "subset"),
        (
            {"train_small": [{"sample_id": "s1"}, {"sample_id": "s1"}]},
            [_row("s1", "v1")],
            [],
            "duplicate",
        ),
        (
            {"train_small": [{"sample_id": "s1"}]},
            [_row("s1", "v1")],
            [_row("s2", "v1")],
            "video group",
        ),
    ],
)
def test_make_training_cohorts_rejects_bad_group_or_subset_integrity(
    manifest, train, validation, message
):
    with pytest.raises(ValueError, match=message):
        make_training_cohorts(manifest, train, validation)


def test_paired_metric_interval_reports_video_row_bootstrap():
    reference = np.asarray([[4.0, 0.0], [0.0, 4.0], [0.0, 4.0]])
    candidate = np.asarray([[4.0, 0.0], [4.0, 0.0], [4.0, 0.0]])
    targets = np.asarray([0, 0, 0])

    result = paired_metric_interval(reference, candidate, targets, repetitions=100, seed=17)

    assert result["delta_accuracy_candidate_minus_reference"] == pytest.approx(2 / 3)
    assert result["resampling_unit"] == "video group; one row per video"
    assert result["bootstrap_repetitions"] == 100
    assert set(result["metrics"]) == {"accuracy", "nll", "brier"}


def test_audio_only_arm_hides_video_tensor_and_keeps_targets():
    train = (object(), object(), object())
    validation = (object(), object(), object())

    audio_train, audio_validation = _data_for_arm("audio_only", train, validation)
    av_train, av_validation = _data_for_arm("pooled_av", train, validation)

    assert audio_train == (None, train[1], train[2])
    assert audio_validation == (None, validation[1], validation[2])
    assert av_train is train
    assert av_validation is validation


def test_effective_config_records_terminal_run_status(tmp_path):
    path = tmp_path / "effective_config.json"
    path.write_text(json.dumps({"status": "RUNNING", "seed": 17}), encoding="utf-8")

    _set_effective_config_status(path, "PASS_FRESH_AV_TEMPORAL_SCALE_FIT")

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "status": "PASS_FRESH_AV_TEMPORAL_SCALE_FIT",
        "seed": 17,
    }
