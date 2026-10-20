from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_holoassist_metadata import sha256_file

torch = pytest.importorskip("torch")

from scripts.train_holoassist_temporal_readout import (  # noqa: E402
    attach_imu_features,
    collate_sessions,
    load_feature_split,
    metric_summary,
    train_arm,
    validate_feature_receipt,
    validate_split_identity,
)


def test_unused_train_textfree_config_matches_split_and_input_policy() -> None:
    config_path = (
        Path(__file__).parents[1]
        / "configs"
        / "pretrained_reuse"
        / "holoassist_textfree_unused_train_v1.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert set(config["encoders"]) == {"video", "audio"}
    assert config["conditioning"] == "task_type_id"
    assert "previously unused official training sessions" in config["validation"]
    assert "official validation and test payloads are not read" in config["validation"]


def test_sensor_fusion_config_freezes_matched_three_arm_policy() -> None:
    config_path = (
        Path(__file__).parents[1]
        / "configs"
        / "pretrained_reuse"
        / "holoassist_av_sensor_temporal_readout_v1.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["sensor_fusion"] is True
    assert config["seed"] == 17
    assert config["arms"] == ["av", "sensor_only", "avs"]
    assert config["max_optimizer_steps"] == 2048
    assert config["evaluation_interval_steps"] == 64
    assert config["early_stopping"] is False
    assert config["best_checkpoint_selector"] == "lowest validation NLL; ties keep earliest step"


def write_features(path, sessions, *, include_text=True):
    rows = []
    for session, count, target_offset in sessions:
        for window in range(count):
            rows.append((session, window, (window + target_offset) % 2))
    arrays = {
        "video": np.zeros((len(rows), 768), dtype=np.float32),
        "audio": np.ones((len(rows), 384), dtype=np.float32),
        "target": np.asarray([row[2] for row in rows], dtype=np.int64),
        "loss_mask": np.ones(len(rows), dtype=np.bool_),
        "sample_id": np.asarray([f"{row[0]}:window-{row[1]}" for row in rows], dtype=str),
        "session_id": np.asarray([row[0] for row in rows], dtype=str),
        "window_index": np.asarray([row[1] for row in rows], dtype=np.int64),
        "media_sha256": np.asarray([f"sha-{row[0]}" for row in rows], dtype=str),
        "task_type": np.asarray(["synthetic task" for _ in rows], dtype=str),
    }
    if include_text:
        arrays["text"] = np.asarray(
            [np.full(384, 2.0 if row[0].startswith("train") else 3.0) for row in rows],
            dtype=np.float32,
        )
    np.savez_compressed(path, **arrays)


def test_split_groups_sessions_and_collates_chronological_windows(tmp_path) -> None:
    train_path = tmp_path / "train.npz"
    validation_path = tmp_path / "validation.npz"
    write_features(train_path, [("train-b", 2, 0), ("train-a", 3, 1)])
    write_features(validation_path, [("validation-c", 2, 0)])
    train = load_feature_split(train_path)
    validation = load_feature_split(validation_path)
    validate_split_identity(train, validation)

    assert train["session_names"].tolist() == ["train-a", "train-b"]
    batch = collate_sessions(train, [0, 1], torch.device("cpu"))
    video, audio, text, lengths, targets, mask = batch
    assert video.shape == (2, 3, 768)
    assert audio.shape == (2, 3, 384)
    assert text.shape == (2, 384)
    assert lengths.tolist() == [3, 2]
    assert targets[1, 2].item() == -1
    assert not mask[1, 2].item()


def test_feature_split_rejects_cross_session_window_gap(tmp_path) -> None:
    path = tmp_path / "broken.npz"
    write_features(path, [("train-a", 2, 0)])
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    values["window_index"][1] = 3
    np.savez_compressed(path, **values)

    with pytest.raises(ValueError, match="non-contiguous"):
        load_feature_split(path)


def test_split_identity_rejects_media_asset_overlap(tmp_path) -> None:
    train_path = tmp_path / "train.npz"
    validation_path = tmp_path / "validation.npz"
    write_features(train_path, [("train-a", 2, 0)])
    write_features(validation_path, [("validation-a", 2, 0)])
    train = load_feature_split(train_path)
    validation = load_feature_split(validation_path)
    validation["media_sha256"][:] = train["media_sha256"][0]

    with pytest.raises(ValueError, match="media_sha256 overlap"):
        validate_split_identity(train, validation)


def test_training_requires_hash_bound_feature_receipt(tmp_path) -> None:
    train_path = tmp_path / "train.npz"
    validation_path = tmp_path / "validation.npz"
    write_features(train_path, [("train-a", 2, 0)])
    write_features(validation_path, [("validation-a", 2, 0)])
    train = load_feature_split(train_path)
    validation = load_feature_split(validation_path)

    def id_hash(data):
        return hashlib.sha256(
            "\n".join(map(str, data["sample_id"])).encode()
        ).hexdigest()

    receipt = {
        "status": "PASS_HOLOASSIST_FROZEN_FEATURE_CACHE",
        "packed_feature_sha256": {
            "train": sha256_file(train_path),
            "validation": sha256_file(validation_path),
        },
        "report": {
            "train": {"ordered_sample_ids_sha256": id_hash(train)},
            "validation": {"ordered_sample_ids_sha256": id_hash(validation)},
        },
        "train_validation_session_overlap": 0,
        "train_validation_media_overlap": 0,
        "test_data_loaded": False,
        "input_feature_banks": {
            modality: {
                "model": {
                    "model_id": modality,
                    "revision": "pinned",
                    "weights_sha256": "0" * 64,
                }
            }
            for modality in ("video", "audio", "text")
        },
    }
    validate_feature_receipt(
        receipt,
        train_path=train_path,
        validation_path=validation_path,
        train=train,
        validation=validation,
        expected_encoders={
            modality: {
                "model_id": modality,
                "revision": "pinned",
                "weights_sha256": "0" * 64,
            }
            for modality in ("video", "audio", "text")
        },
    )
    receipt["packed_feature_sha256"]["validation"] = "0" * 64
    with pytest.raises(ValueError, match="receipt hashes"):
        validate_feature_receipt(
            receipt,
            train_path=train_path,
            validation_path=validation_path,
            train=train,
            validation=validation,
            expected_encoders={
                modality: {
                    "model_id": modality,
                    "revision": "pinned",
                    "weights_sha256": "0" * 64,
                }
                for modality in ("video", "audio", "text")
            },
        )


def test_metrics_are_finite_and_use_labelled_examples() -> None:
    result = metric_summary(torch.tensor([[4.0, 0.0], [0.0, 4.0]]), torch.tensor([0, 1]))
    assert result["count"] == 2
    assert result["accuracy"] == 1.0
    assert result["sensitivity"] == 1.0
    assert result["specificity"] == 1.0
    assert result["balanced_accuracy"] == 1.0
    assert result["positive_prevalence"] == 0.5
    assert result["nll"] < 0.02
    assert result["brier"] < 0.01
    assert result["ece"] < 0.02


def test_fixed_budget_run_keeps_best_and_final_checkpoints_separate(tmp_path) -> None:
    train_path = tmp_path / "train.npz"
    validation_path = tmp_path / "validation.npz"
    write_features(train_path, [("train-a", 2, 0), ("train-b", 2, 1)])
    write_features(validation_path, [("validation-a", 2, 0)])
    train = load_feature_split(train_path)
    validation = load_feature_split(validation_path)
    validate_split_identity(train, validation)
    config = {
        "seed": 17,
        "hidden_size": 8,
        "batch_sessions": 1,
        "max_optimizer_steps": 2,
        "evaluation_interval_steps": 1,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "ece_bins": 5,
    }

    result = train_arm(
        arm="joint",
        train=train,
        validation=validation,
        config=config,
        device=torch.device("cpu"),
        output=tmp_path / "result",
    )

    assert result["optimizer_steps"] == 2
    assert result["early_stopping"] is False
    assert result["best_reload_max_probability_delta"] <= 1e-6
    assert result["final_validation"]["by_task_type"]["synthetic task"]["count"] == 2
    assert set(result["final_validation"]["task_type_macro"]) == {
        "accuracy",
        "nll",
        "brier",
        "ece",
    }
    assert (tmp_path / "result" / "best-checkpoint.pt").is_file()
    assert (tmp_path / "result" / "final-checkpoint.pt").is_file()
    assert (tmp_path / "result" / "training-history.json").is_file()


def test_task_type_conditioned_run_trains_without_text_features(tmp_path) -> None:
    train_path = tmp_path / "train.npz"
    validation_path = tmp_path / "validation.npz"
    write_features(train_path, [("train-a", 2, 0), ("train-b", 2, 1)], include_text=False)
    write_features(validation_path, [("validation-a", 2, 0)], include_text=False)
    train = load_feature_split(train_path, require_text=False)
    validation = load_feature_split(validation_path, require_text=False)
    config = {
        "seed": 17,
        "hidden_size": 8,
        "batch_sessions": 1,
        "max_optimizer_steps": 2,
        "evaluation_interval_steps": 1,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "ece_bins": 5,
    }

    result = train_arm(
        arm="joint",
        train=train,
        validation=validation,
        config=config,
        device=torch.device("cpu"),
        output=tmp_path / "task-conditioned",
        task_type_to_id={"synthetic task": 0},
    )

    assert result["optimizer_steps"] == 2
    assert result["text_features_used"] is False
    assert result["task_type_vocabulary"] == {"synthetic task": 0}
    assert result["best_reload_max_probability_delta"] <= 1e-6


def test_imu_features_require_exact_sample_order_and_metadata(tmp_path) -> None:
    av_path = tmp_path / "train.npz"
    write_features(av_path, [("train-a", 2, 0)], include_text=False)
    data = load_feature_split(av_path, require_text=False)
    sequence = np.ones((len(data["sample_id"]), 30, 21), dtype=np.float32)
    metadata = [
        {
            "split": "train",
            "id": str(data["sample_id"][index]),
            "session_id": str(data["session_id"][index]),
            "window_index": int(data["window_index"][index]),
            "target": int(data["target"][index]),
            "loss_mask": bool(data["loss_mask"][index]),
            "media_sha256": str(data["media_sha256"][index]),
            "task_type": str(data["task_type"][index]),
        }
        for index in range(len(data["sample_id"]))
    ]
    imu_path = tmp_path / "imu.npz"
    np.savez_compressed(
        imu_path,
        sample_id=data["sample_id"],
        features=sequence,
        flattened_features=sequence.reshape(len(sequence), -1),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    id_hash = hashlib.sha256("\n".join(map(str, data["sample_id"])).encode()).hexdigest()
    receipt = {
        "splits": {
            "train": {
                "sha256": sha256_file(imu_path),
                "sample_count": len(data["sample_id"]),
                "unique_sample_id_order_sha256": id_hash,
            }
        }
    }

    joined = attach_imu_features(data, imu_path, split="train", receipt=receipt)
    assert joined["sensor"].shape == (2, 630)

    validation_av_path = tmp_path / "validation.npz"
    write_features(validation_av_path, [("validation-a", 2, 0)], include_text=False)
    validation = load_feature_split(validation_av_path, require_text=False)
    validation_sequence = np.full((2, 30, 21), 2.0, dtype=np.float32)
    validation_metadata = [
        {
            "split": "validation",
            "id": str(validation["sample_id"][index]),
            "session_id": str(validation["session_id"][index]),
            "window_index": int(validation["window_index"][index]),
            "target": int(validation["target"][index]),
            "loss_mask": bool(validation["loss_mask"][index]),
            "media_sha256": str(validation["media_sha256"][index]),
            "task_type": str(validation["task_type"][index]),
        }
        for index in range(len(validation["sample_id"]))
    ]
    validation_imu_path = tmp_path / "validation-imu.npz"
    np.savez_compressed(
        validation_imu_path,
        sample_id=validation["sample_id"],
        features=validation_sequence,
        flattened_features=validation_sequence.reshape(len(validation_sequence), -1),
        metadata_json=np.asarray(json.dumps(validation_metadata)),
    )
    validation_id_hash = hashlib.sha256(
        "\n".join(map(str, validation["sample_id"])).encode()
    ).hexdigest()
    receipt["splits"]["validation"] = {
        "sha256": sha256_file(validation_imu_path),
        "sample_count": len(validation["sample_id"]),
        "unique_sample_id_order_sha256": validation_id_hash,
    }
    validation = attach_imu_features(
        validation,
        validation_imu_path,
        split="validation",
        receipt=receipt,
    )
    config = {
        "seed": 17,
        "hidden_size": 8,
        "batch_sessions": 1,
        "max_optimizer_steps": 1,
        "evaluation_interval_steps": 1,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "ece_bins": 5,
    }
    for arm in ("av", "sensor_only", "avs"):
        result = train_arm(
            arm=arm,
            train=joined,
            validation=validation,
            config=config,
            device=torch.device("cpu"),
            output=tmp_path / f"{arm}-run",
            task_type_to_id={"synthetic task": 0},
        )
        assert result["optimizer_steps"] == 1
        assert result["sensor_features_used"] is (arm != "av")
        assert 0 < result["parameters_with_gradients"] <= result["trainable_parameters"]
        assert result["best_reload_max_probability_delta"] <= 1e-6

    metadata[0]["media_sha256"] = "wrong-asset"
    np.savez_compressed(
        imu_path,
        sample_id=data["sample_id"],
        features=sequence,
        flattened_features=sequence.reshape(len(sequence), -1),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    receipt["splits"]["train"]["sha256"] = sha256_file(imu_path)
    with pytest.raises(ValueError, match="metadata differs"):
        attach_imu_features(data, imu_path, split="train", receipt=receipt)
