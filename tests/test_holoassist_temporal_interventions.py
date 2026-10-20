from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from scripts.audit_holoassist_metadata import sha256_file

torch = pytest.importorskip("torch")

from scripts.evaluate_holoassist_temporal_interventions import (  # noqa: E402
    evaluate_intervention,
    run_interventions,
)
from scripts.train_holoassist_temporal_readout import (  # noqa: E402
    _evaluate,
    load_feature_split,
)
from tiny_omni_decision.temporal_decision import (  # noqa: E402
    CausalGatedTemporalDecision,
    TaskConditionedTemporalDecision,
)
from tiny_omni_decision.temporal_interventions import (  # noqa: E402
    apply_feature_intervention,
)


def _write_features(path, sessions):
    rows = []
    for session, length, task_type, offset in sessions:
        for window in range(length):
            rows.append((session, window, task_type, offset + window))
    count = len(rows)
    np.savez_compressed(
        path,
        video=np.stack([np.full(768, row[3], dtype=np.float32) for row in rows]),
        audio=np.stack([np.full(384, row[3] + 100, dtype=np.float32) for row in rows]),
        target=np.asarray([row[1] % 2 for row in rows], dtype=np.int64),
        loss_mask=np.ones(count, dtype=np.bool_),
        sample_id=np.asarray([f"{row[0]}:w{row[1]}" for row in rows], dtype=str),
        session_id=np.asarray([row[0] for row in rows], dtype=str),
        window_index=np.asarray([row[1] for row in rows], dtype=np.int64),
        media_sha256=np.asarray([f"hash-{row[0]}" for row in rows], dtype=str),
        task_type=np.asarray([row[2] for row in rows], dtype=str),
    )


def _features(tmp_path):
    path = tmp_path / "validation.npz"
    _write_features(
        path,
        [
            ("a", 2, "cut", 1),
            ("b", 2, "cut", 10),
            ("c", 3, "cut", 20),
            ("d", 2, "stir", 30),
        ],
    )
    return load_feature_split(path, require_text=False)


@pytest.mark.parametrize("intervention", ["baseline", "audio_missing", "video_missing"])
def test_task_conditioned_decision_supports_true_modality_presence_masks(intervention):
    torch.manual_seed(17)
    model = TaskConditionedTemporalDecision(task_count=2, hidden_size=8).eval()
    video = torch.randn(2, 3, 768)
    audio = torch.randn(2, 3, 384)
    task_ids = torch.tensor([0, 1])
    lengths = torch.tensor([3, 2])
    video_present = torch.ones((2, 3), dtype=torch.bool)
    audio_present = torch.ones_like(video_present)
    if intervention == "audio_missing":
        audio_present.zero_()
    elif intervention == "video_missing":
        video_present.zero_()

    output = model(
        video,
        audio,
        task_ids,
        lengths=lengths,
        video_present=video_present,
        audio_present=audio_present,
    )
    assert output.logits.shape == (2, 3, 2)
    assert torch.isfinite(output.logits).all()
    if intervention == "audio_missing":
        assert torch.count_nonzero(output.modality_weights[..., 1]) == 0
        assert torch.all(output.modality_weights[..., 0][0, :3] == 1)
    if intervention == "video_missing":
        assert torch.count_nonzero(output.modality_weights[..., 0]) == 0
        assert torch.all(output.modality_weights[..., 1][0, :3] == 1)


def test_task_conditioned_mask_defaults_preserve_existing_outputs():
    torch.manual_seed(17)
    model = TaskConditionedTemporalDecision(task_count=2, hidden_size=8).eval()
    video = torch.randn(2, 3, 768)
    audio = torch.randn(2, 3, 384)
    task_ids = torch.tensor([0, 1])
    lengths = torch.tensor([3, 2])
    default = model(video, audio, task_ids, lengths=lengths)
    explicit = model(
        video,
        audio,
        task_ids,
        lengths=lengths,
        video_present=torch.ones((2, 3), dtype=torch.bool),
        audio_present=torch.ones((2, 3), dtype=torch.bool),
    )
    torch.testing.assert_close(default.logits, explicit.logits, rtol=0, atol=0)
    torch.testing.assert_close(default.modality_weights, explicit.modality_weights, rtol=0, atol=0)


def test_task_conditioned_mask_rejects_steps_with_no_modality():
    model = TaskConditionedTemporalDecision(task_count=1, hidden_size=8)
    present = torch.zeros((1, 2), dtype=torch.bool)
    with pytest.raises(ValueError, match="at least one available modality"):
        model(
            torch.zeros(1, 2, 768),
            torch.zeros(1, 2, 384),
            torch.zeros(1, dtype=torch.long),
            video_present=present,
            audio_present=present,
        )


def test_single_modality_mode_does_not_mutate_caller_presence_masks():
    model = TaskConditionedTemporalDecision(task_count=1, hidden_size=8, modality_mode="video_only")
    video_present = torch.ones((1, 2), dtype=torch.bool)
    audio_present = torch.ones_like(video_present)
    model(
        torch.zeros(1, 2, 768),
        torch.zeros(1, 2, 384),
        torch.zeros(1, dtype=torch.long),
        video_present=video_present,
        audio_present=audio_present,
    )
    assert video_present.all()
    assert audio_present.all()


def test_causal_single_modality_mode_does_not_mutate_caller_presence_masks():
    model = CausalGatedTemporalDecision(hidden_size=8, modality_mode="audio_only")
    video_present = torch.ones((1, 2), dtype=torch.bool)
    audio_present = torch.ones_like(video_present)
    model(
        torch.zeros(1, 2, 768),
        torch.zeros(1, 2, 384),
        torch.zeros(1, 384),
        video_present=video_present,
        audio_present=audio_present,
    )
    assert video_present.all()
    assert audio_present.all()


@pytest.mark.parametrize(
    ("intervention", "expected_eligible"),
    [
        ("baseline", 0),
        ("audio_missing", 0),
        ("video_missing", 0),
        ("audio_reverse_time", 0),
        ("video_reverse_time", 0),
        ("audio_shuffle_session", 2),
    ],
)
def test_feature_interventions_are_deterministic_and_preserve_identity(
    tmp_path, intervention, expected_eligible
):
    data = _features(tmp_path)
    original_audio = data["audio"].copy()
    original_video = data["video"].copy()
    original_ids = data["sample_id"].copy()

    first = apply_feature_intervention(data, intervention, seed=17)
    second = apply_feature_intervention(data, intervention, seed=17)

    assert first.eligible_session_count == expected_eligible
    np.testing.assert_array_equal(first.video, second.video)
    np.testing.assert_array_equal(first.audio, second.audio)
    np.testing.assert_array_equal(first.video_present, second.video_present)
    np.testing.assert_array_equal(first.audio_present, second.audio_present)
    np.testing.assert_array_equal(data["audio"], original_audio)
    np.testing.assert_array_equal(data["video"], original_video)
    np.testing.assert_array_equal(data["sample_id"], original_ids)
    if intervention == "audio_missing":
        assert not first.audio_present.any()
        assert np.count_nonzero(first.audio) == 0
        assert first.video_present.all()
    elif intervention == "video_missing":
        assert not first.video_present.any()
        assert np.count_nonzero(first.video) == 0
        assert first.audio_present.all()
    elif intervention == "audio_reverse_time":
        session = data["session_indices"][0]
        np.testing.assert_array_equal(first.audio[session], original_audio[session[::-1]])
    elif intervention == "video_reverse_time":
        session = data["session_indices"][0]
        np.testing.assert_array_equal(first.video[session], original_video[session[::-1]])
    elif intervention == "audio_shuffle_session":
        eligible = ["a", "b"]
        a, b = (data["session_indices"][data["session_names"].tolist().index(x)] for x in eligible)
        np.testing.assert_array_equal(first.audio[a], original_audio[b])
        np.testing.assert_array_equal(first.audio[b], original_audio[a])
        assert first.shuffle_group_count == 1
        assert np.array_equal(first.video, original_video)


def test_audio_shuffle_excludes_different_task_types_even_when_lengths_match(tmp_path):
    data = _features(tmp_path)
    result = apply_feature_intervention(data, "audio_shuffle_session", seed=17)

    session_to_task = {
        str(name): str(data["task_type"][indices[0]])
        for name, indices in zip(data["session_names"], data["session_indices"], strict=True)
    }
    assert session_to_task["d"] == "stir"
    d_index = data["session_names"].tolist().index("d")
    np.testing.assert_array_equal(
        result.audio[data["session_indices"][d_index]],
        data["audio"][data["session_indices"][d_index]],
    )


def test_evaluator_reports_fixed_validation_predictions_for_each_intervention(tmp_path):
    torch.manual_seed(17)
    model = TaskConditionedTemporalDecision(task_count=2, hidden_size=8)
    data = _features(tmp_path)
    task_ids = {"cut": 0, "stir": 1}
    observed_ids = []
    for intervention in (
        "baseline",
        "audio_missing",
        "video_missing",
        "audio_reverse_time",
        "video_reverse_time",
        "audio_shuffle_session",
    ):
        metrics, predictions, details = evaluate_intervention(
            model,
            data,
            task_ids,
            device=torch.device("cpu"),
            batch_sessions=2,
            ece_bins=5,
            intervention=intervention,
            seed=17,
        )
        assert metrics["count"] == 9
        assert set(metrics["by_task_type"]) == {"cut", "stir"}
        assert set(metrics["mean_gate_weights"]) == {"video", "audio"}
        assert len(predictions) == 9
        assert set(prediction["id"] for prediction in predictions) == set(
            map(str, data["sample_id"])
        )
        assert all(np.isfinite(prediction["probabilities"]).all() for prediction in predictions)
        observed_ids.append([prediction["id"] for prediction in predictions])
        if intervention == "audio_shuffle_session":
            assert details == {"eligible_session_count": 2, "shuffle_group_count": 1}
        elif intervention == "audio_missing":
            assert metrics["mean_gate_weights"]["audio"] == 0.0
        elif intervention == "video_missing":
            assert metrics["mean_gate_weights"]["video"] == 0.0
    assert all(ids == observed_ids[0] for ids in observed_ids[1:])


def test_intervention_baseline_matches_training_validation_path(tmp_path):
    torch.manual_seed(23)
    model = TaskConditionedTemporalDecision(task_count=2, hidden_size=8).eval()
    data = _features(tmp_path)
    task_ids = {"cut": 0, "stir": 1}
    expected_metrics, expected_predictions = _evaluate(
        model,
        data,
        torch.device("cpu"),
        batch_sessions=2,
        ece_bins=5,
        task_type_to_id=task_ids,
    )
    actual_metrics, actual_predictions, _details = evaluate_intervention(
        model,
        data,
        task_ids,
        device=torch.device("cpu"),
        batch_sessions=2,
        ece_bins=5,
        intervention="baseline",
        seed=17,
    )

    actual_metric_core = {
        key: value for key, value in actual_metrics.items() if key != "mean_gate_weights"
    }
    actual_metric_core["by_task_type"] = {
        task: {key: value for key, value in metrics.items() if key != "mean_gate_weights"}
        for task, metrics in actual_metrics["by_task_type"].items()
    }
    assert actual_metric_core == expected_metrics
    for expected, actual in zip(expected_predictions, actual_predictions, strict=True):
        assert {key: expected[key] for key in expected if key != "probabilities"} == {
            key: actual[key] for key in actual if key != "probabilities"
        }
        np.testing.assert_allclose(
            actual["probabilities"], expected["probabilities"], rtol=0, atol=0
        )


def test_intervention_runner_checks_run_hashes_and_writes_separate_outputs(tmp_path):
    run_dir = tmp_path / "readout-run"
    joint_dir = run_dir / "joint"
    joint_dir.mkdir(parents=True)
    train_path = tmp_path / "train.npz"
    validation_path = tmp_path / "validation.npz"
    _write_features(train_path, [("train-a", 2, "cut", 1), ("train-b", 2, "stir", 10)])
    _write_features(
        validation_path,
        [("val-a", 2, "cut", 20), ("val-b", 2, "cut", 30), ("val-c", 2, "stir", 40)],
    )
    train = load_feature_split(train_path, require_text=False)
    validation = load_feature_split(validation_path, require_text=False)
    config = {
        "conditioning": "task_type_id",
        "arms": ["video_only", "audio_only", "joint"],
        "hidden_size": 8,
        "batch_sessions": 2,
        "ece_bins": 5,
        "encoders": {
            modality: {
                "model_id": modality,
                "revision": "revision-1",
                "weights_sha256": "0" * 64,
            }
            for modality in ("video", "audio")
        },
    }
    feature_receipt = {
        "status": "PASS_HOLOASSIST_FROZEN_FEATURE_CACHE",
        "packed_feature_sha256": {
            "train": sha256_file(train_path),
            "validation": sha256_file(validation_path),
        },
        "report": {
            split: {
                "ordered_sample_ids_sha256": hashlib.sha256(
                    "\n".join(map(str, values["sample_id"])).encode()
                ).hexdigest()
            }
            for split, values in (("train", train), ("validation", validation))
        },
        "train_validation_session_overlap": 0,
        "train_validation_media_overlap": 0,
        "test_data_loaded": False,
        "input_feature_banks": {
            modality: {
                "model": {
                    "model_id": modality,
                    "revision": "revision-1",
                    "weights_sha256": "0" * 64,
                }
            }
            for modality in ("video", "audio")
        },
    }
    feature_receipt_path = tmp_path / "receipt.json"
    feature_receipt_path.write_text(json.dumps(feature_receipt), encoding="utf-8")
    task_vocab = {"cut": 0, "stir": 1}
    model = TaskConditionedTemporalDecision(task_count=2, hidden_size=8).eval()
    checkpoint_path = joint_dir / "best-checkpoint.pt"
    torch.save(model.state_dict(), checkpoint_path)
    baseline_metrics, baseline_predictions = _evaluate(
        model,
        validation,
        torch.device("cpu"),
        batch_sessions=2,
        ece_bins=5,
        task_type_to_id=task_vocab,
    )
    best_validation_nll = float(baseline_metrics["nll"])
    (joint_dir / "best-checkpoint.json").write_text(
        json.dumps({"step": 0, "validation_nll": best_validation_nll}),
        encoding="utf-8",
    )
    (joint_dir / "predictions-step-000000.json").write_text(
        json.dumps(baseline_predictions), encoding="utf-8"
    )
    (joint_dir / "result.json").write_text(
        json.dumps(
            {
                "arm": "joint",
                "best_step": 0,
                "best_validation_nll": best_validation_nll,
                "best_checkpoint_sha256": sha256_file(checkpoint_path),
            }
        ),
        encoding="utf-8",
    )
    run_metadata = {
        "status": "complete",
        "sealed_audit_loaded": False,
        "config": config,
        "results": [{"arm": arm} for arm in ("video_only", "audio_only", "joint")],
        "task_type_vocabulary": task_vocab,
        "train_features_sha256": sha256_file(train_path),
        "validation_features_sha256": sha256_file(validation_path),
        "feature_receipt_sha256": sha256_file(feature_receipt_path),
    }
    (run_dir / "run-metadata.json").write_text(json.dumps(run_metadata), encoding="utf-8")

    output_dir = tmp_path / "intervention-attempt"
    result = run_interventions(
        run_dir=run_dir,
        train_features=train_path,
        validation_features=validation_path,
        feature_receipt_path=feature_receipt_path,
        output_dir=output_dir,
        device_name="cpu",
        seed=17,
    )

    assert result["status"] == "complete_fixed_validation_interventions"
    assert result["sealed_audit_loaded"] is False
    assert result["results"]["baseline"]["saved_checkpoint_replay_max_probability_delta"] == 0
    assert result["results"]["audio_shuffle_session"]["eligible_session_count"] == 2
    assert (output_dir / "run-metadata.json").is_file()
    assert (
        json.loads((output_dir / "run-metadata.json").read_text())["sealed_audit_loaded"] is False
    )
    assert len(list(output_dir.glob("predictions-*.json"))) == 6
