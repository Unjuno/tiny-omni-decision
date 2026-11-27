import json
import sys

import numpy as np
import pytest

from scripts.audit_videoprism_checkpoint import CHECKPOINT_SHA256, MODEL_REVISION
from scripts.evaluate_intphys_next_feature_predictor import (
    load_dev_features,
    load_split_identities,
    main,
    sha256_file,
    validate_cross_split_identities,
    validate_receipts,
)
from scripts.extract_frozen_videoprism_features import SOURCE_COMMIT as VIDEOPRISM_SOURCE_COMMIT
from scripts.extract_intphys_dev_features import (
    EXPECTED_ARCHIVE_BYTES as EXPECTED_DEV_BYTES,
)
from scripts.extract_intphys_dev_features import (
    EXPECTED_ARCHIVE_MD5 as EXPECTED_DEV_MD5,
)
from scripts.extract_intphys_dev_features import (
    EXPECTED_ARCHIVE_SHA256 as EXPECTED_DEV_SHA256,
)
from scripts.extract_intphys_train_features import (
    EXPECTED_ARCHIVE_BYTES as EXPECTED_TRAIN_BYTES,
)
from scripts.extract_intphys_train_features import (
    EXPECTED_ARCHIVE_MD5 as EXPECTED_TRAIN_MD5,
)


def test_load_dev_features_requires_complete_balanced_scene_groups(tmp_path):
    path = tmp_path / "dev-features.npz"
    np.savez_compressed(
        path,
        features=np.zeros((4, 8, 3), dtype=np.float32),
        sample_ids=np.asarray(["a", "b", "c", "d"]),
        group_ids=np.asarray(["g", "g", "g", "g"]),
        blocks=np.asarray(["O1"] * 4),
        labels=np.asarray([0, 1, 0, 1], dtype=np.int8),
        normalized_media_sha256=np.asarray([f"media-{i}" for i in range(4)]),
    )
    features, sample_ids, group_ids, blocks, labels, media_hashes = load_dev_features(path)
    assert features.shape == (4, 8, 3)
    assert len(sample_ids) == len(group_ids) == len(blocks) == len(labels) == 4
    assert len(media_hashes) == 4

    np.savez_compressed(
        path,
        features=np.zeros((4, 8, 3), dtype=np.float32),
        sample_ids=np.asarray(["a", "b", "c", "d"]),
        group_ids=np.asarray(["g"] * 4),
        blocks=np.asarray(["O1"] * 4),
        labels=np.asarray([0, 0, 0, 1], dtype=np.int8),
        normalized_media_sha256=np.asarray([f"media-{i}" for i in range(4)]),
    )
    with pytest.raises(ValueError, match="2/2 label group"):
        load_dev_features(path)


def test_load_dev_features_rejects_nan_and_duplicate_ids(tmp_path):
    path = tmp_path / "bad.npz"
    for features, ids in (
        (np.full((4, 8, 3), np.nan, dtype=np.float32), ["a", "b", "c", "d"]),
        (np.zeros((4, 8, 3), dtype=np.float32), ["a", "a", "c", "d"]),
    ):
        np.savez_compressed(
            path,
            features=features,
            sample_ids=np.asarray(ids),
            group_ids=np.asarray(["g"] * 4),
            blocks=np.asarray(["O1"] * 4),
            labels=np.asarray([0, 1, 0, 1], dtype=np.int8),
            normalized_media_sha256=np.asarray([f"media-{i}" for i in range(4)]),
        )
        with pytest.raises(ValueError, match="invalid finite dev arrays"):
            load_dev_features(path)


def test_split_identity_loader_requires_normalized_media_hashes(tmp_path):
    path = tmp_path / "train.npz"
    np.savez_compressed(
        path,
        sample_ids=np.asarray(["a", "b"]),
        group_ids=np.asarray(["g1", "g2"]),
        sampled_rgb_pixels_sha256=np.asarray(["m1", "m2"]),
    )
    assert load_split_identities(path) == (["a", "b"], ["g1", "g2"], ["m1", "m2"])


def test_cross_split_identity_gate_rejects_normalized_media_overlap():
    no_overlap = {
        "train": (["t"], ["tg"], ["tm"]),
        "validation": (["v"], ["vg"], ["vm"]),
        "dev": (["d"], ["dg"], ["dm"]),
    }
    assert validate_cross_split_identities(no_overlap) == {
        "sample_id": 0,
        "group_id": 0,
        "sampled_media_sha256": 0,
    }
    overlap = {
        **no_overlap,
        "dev": (["d"], ["dg"], ["tm"]),
    }
    with pytest.raises(ValueError, match="sampled_media_sha256 overlap between train and dev"):
        validate_cross_split_identities(overlap)


def test_evaluation_receipts_fail_closed_on_checkpoint_or_feature_hash_mismatch(tmp_path):
    checkpoint = tmp_path / "best.pt"
    training_metadata_path = tmp_path / "training.json"
    corpus_report_path = tmp_path / "corpus.json"
    train_features = tmp_path / "train.npz"
    validation_features = tmp_path / "validation.npz"
    dev_features = tmp_path / "dev.npz"
    dev_report_path = tmp_path / "dev-report.json"
    checkpoint.write_bytes(b"best checkpoint")
    train_features.write_bytes(b"train cache")
    validation_features.write_bytes(b"validation cache")
    dev_features.write_bytes(b"dev cache")

    training_metadata_path.write_text(
        json.dumps(
            {
                "status": "complete_train_only_next_feature_predictor",
                "sequence_length": 8,
                "best_checkpoint_sha256": sha256_file(checkpoint),
                "train_features_sha256": sha256_file(train_features),
                "validation_features_sha256": sha256_file(validation_features),
            }
        ),
        encoding="utf-8",
    )
    corpus_report_path.write_text(
        json.dumps(
            {
                "status": "complete_intphys_train_frozen_feature_extraction",
                "all_train_labels_possible": True,
                "archive_bytes": EXPECTED_TRAIN_BYTES,
                "archive_md5": EXPECTED_TRAIN_MD5,
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "model_revision": MODEL_REVISION,
                "videoprism_source_commit": VIDEOPRISM_SOURCE_COMMIT,
                "feature_shape": [8, 768],
                "feature_archive_sha256": {
                    "train": sha256_file(train_features),
                    "validation": sha256_file(validation_features),
                },
            }
        ),
        encoding="utf-8",
    )
    dev_report_path.write_text(
        json.dumps(
            {
                "status": "complete_intphys_dev_frozen_features_for_one_shot_diagnostic",
                "archive_bytes": EXPECTED_DEV_BYTES,
                "archive_md5": EXPECTED_DEV_MD5,
                "archive_sha256": EXPECTED_DEV_SHA256,
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "model_revision": MODEL_REVISION,
                "videoprism_source_commit": VIDEOPRISM_SOURCE_COMMIT,
                "feature_shape": [8, 768],
                "feature_file_sha256": sha256_file(dev_features),
                "sealed_test_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    validate_receipts(
        checkpoint=checkpoint,
        train_metadata_path=training_metadata_path,
        train_corpus_report_path=corpus_report_path,
        train_features_path=train_features,
        validation_features_path=validation_features,
        dev_features_path=dev_features,
        dev_report_path=dev_report_path,
    )

    checkpoint.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="checkpoint hash"):
        validate_receipts(
            checkpoint=checkpoint,
            train_metadata_path=training_metadata_path,
            train_corpus_report_path=corpus_report_path,
            train_features_path=train_features,
            validation_features_path=validation_features,
            dev_features_path=dev_features,
            dev_report_path=dev_report_path,
        )

    checkpoint.write_bytes(b"best checkpoint")
    corpus_receipt = json.loads(corpus_report_path.read_text(encoding="utf-8"))
    corpus_receipt["model_revision"] = "un-pinned-revision"
    corpus_report_path.write_text(json.dumps(corpus_receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen encoder"):
        validate_receipts(
            checkpoint=checkpoint,
            train_metadata_path=training_metadata_path,
            train_corpus_report_path=corpus_report_path,
            train_features_path=train_features,
            validation_features_path=validation_features,
            dev_features_path=dev_features,
            dev_report_path=dev_report_path,
        )


def test_fixed_evaluation_cli_writes_grouped_scores_without_probability_claims(
    tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    from tiny_omni_decision.temporal_prediction import NextFrameFeaturePredictor

    rng = np.random.default_rng(17)
    train_features = tmp_path / "train.npz"
    validation_features = tmp_path / "validation.npz"
    dev_features = tmp_path / "dev.npz"
    np.savez_compressed(
        train_features,
        features=rng.normal(size=(4, 8, 6)).astype(np.float32),
        sample_ids=np.asarray([f"train-{i}" for i in range(4)]),
        group_ids=np.asarray([f"train-group-{i}" for i in range(4)]),
        sampled_rgb_pixels_sha256=np.asarray([f"train-media-{i}" for i in range(4)]),
    )
    np.savez_compressed(
        validation_features,
        features=rng.normal(size=(4, 8, 6)).astype(np.float32),
        sample_ids=np.asarray([f"validation-{i}" for i in range(4)]),
        group_ids=np.asarray([f"validation-group-{i}" for i in range(4)]),
        sampled_rgb_pixels_sha256=np.asarray([f"validation-media-{i}" for i in range(4)]),
    )
    dev_count = 16
    np.savez_compressed(
        dev_features,
        features=rng.normal(size=(dev_count, 8, 6)).astype(np.float32),
        sample_ids=np.asarray([f"dev-{i}" for i in range(dev_count)]),
        group_ids=np.asarray([f"dev-group-{i // 4}" for i in range(dev_count)]),
        blocks=np.asarray(["O1"] * dev_count),
        labels=np.tile(np.asarray([0, 1, 0, 1], dtype=np.int8), 4),
        normalized_media_sha256=np.asarray([f"dev-media-{i}" for i in range(dev_count)]),
    )
    checkpoint_path = tmp_path / "best.pt"
    model = NextFrameFeaturePredictor(feature_width=6, hidden_width=4)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_width": 6,
            "hidden_width": 4,
            "best_epoch": 1,
        },
        checkpoint_path,
    )
    corpus_report = {
        "status": "complete_intphys_train_frozen_feature_extraction",
        "all_train_labels_possible": True,
        "archive_bytes": EXPECTED_TRAIN_BYTES,
        "archive_md5": EXPECTED_TRAIN_MD5,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "model_revision": MODEL_REVISION,
        "videoprism_source_commit": VIDEOPRISM_SOURCE_COMMIT,
            "feature_shape": [8, 6],
        "feature_archive_sha256": {
            "train": sha256_file(train_features),
            "validation": sha256_file(validation_features),
        },
    }
    corpus_report_path = tmp_path / "corpus-report.json"
    corpus_report_path.write_text(json.dumps(corpus_report), encoding="utf-8")
    training_metadata = {
        "status": "complete_train_only_next_feature_predictor",
            "sequence_length": 8,
        "best_checkpoint_sha256": sha256_file(checkpoint_path),
        "train_features_sha256": sha256_file(train_features),
        "validation_features_sha256": sha256_file(validation_features),
        "selected_epoch": 1,
    }
    training_metadata_path = tmp_path / "training.json"
    training_metadata_path.write_text(json.dumps(training_metadata), encoding="utf-8")
    dev_report_path = tmp_path / "dev-report.json"
    dev_report_path.write_text(
        json.dumps(
            {
                "status": "complete_intphys_dev_frozen_features_for_one_shot_diagnostic",
                "archive_bytes": EXPECTED_DEV_BYTES,
                "archive_md5": EXPECTED_DEV_MD5,
                "archive_sha256": EXPECTED_DEV_SHA256,
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "model_revision": MODEL_REVISION,
                "videoprism_source_commit": VIDEOPRISM_SOURCE_COMMIT,
                    "feature_shape": [8, 6],
                "feature_file_sha256": sha256_file(dev_features),
                "sealed_test_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "evaluation"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_intphys_next_feature_predictor.py",
            "--checkpoint",
            str(checkpoint_path),
            "--training-metadata",
            str(training_metadata_path),
            "--training-corpus-report",
            str(corpus_report_path),
            "--train-features",
            str(train_features),
            "--validation-features",
            str(validation_features),
            "--dev-features",
            str(dev_features),
            "--dev-report",
            str(dev_report_path),
            "--output-dir",
            str(output_dir),
            "--bootstrap-replicates",
            "100",
        ],
    )

    main()

    report = json.loads((output_dir / "evaluation-report.json").read_text(encoding="utf-8"))
    prediction_rows = [
        json.loads(line)
        for line in (output_dir / "dev-predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert report["status"] == "complete_single_fixed_intphys_dev_diagnostic"
    assert report["dev_scene_groups"] == 4
    assert report["scores_are_calibrated_class_probabilities"] is False
    assert len(prediction_rows) == dev_count
    assert isinstance(prediction_rows[0]["learned_lstm_predicted_impossible"], bool)
