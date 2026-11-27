import hashlib
import json

import pytest

from scripts.compare_intphys_frame_candidates import compare_runs


def _write_prediction_run(directory, frame_count, suffix, *, alter_first_sample=False):
    directory.mkdir()
    rows = []
    for group_index in range(3):
        for item_index, label in enumerate((0, 1, 0, 1)):
            sample_id = f"sample-{group_index}-{item_index}"
            if alter_first_sample and group_index == 0 and item_index == 0:
                sample_id = "different-sample"
            rows.append(
                {
                    "sample_id": sample_id,
                    "scene_group_id": f"scene-{group_index}",
                    "block": "O1",
                    "label_impossible": label,
                    "learned_lstm_anomaly_score": float(label) + 0.1 * item_index + suffix,
                }
            )
    (directory / "dev-predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (directory / "evaluation-report.json").write_text(
        json.dumps(
            {
                "status": "complete_single_fixed_intphys_dev_diagnostic",
                "sealed_test_accessed": False,
                "dev_used_for_fit_or_calibration": False,
                "dev_video_count": len(rows),
                "learned_lstm": {"threshold": 0.5},
            }
        ),
        encoding="utf-8",
    )


def _write_training_metadata(path, frame_count):
    path.write_text(
        json.dumps(
            {
                "sequence_length": frame_count,
                "selected_epoch": 4,
                "best_checkpoint_sha256": f"checkpoint-{frame_count}",
                "seed": 17,
                "max_epochs": 100,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "batch_size": 64,
                "early_stopping_patience": 8,
                "min_delta": 0.0001,
                "optimizer": "AdamW",
                "parameter_count": 264960,
            }
        ),
        encoding="utf-8",
    )


def _write_corpus_report(path, frame_count):
    path.write_text(
        json.dumps(
            {
                "feature_shape": [frame_count, 6],
                "archive_sha256": "same-pinned-source",
                "checkpoint_sha256": "same-pinned-encoder",
                "model_revision": "same-model-revision",
                "videoprism_source_commit": "same-source-commit",
            }
        ),
        encoding="utf-8",
    )


def _bind_evaluation_receipts(evaluation_dir, metadata_path, corpus_path, frame_count):
    report_path = evaluation_dir / "evaluation-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update(
        {
            "training_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            "training_corpus_report_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
            "checkpoint_sha256": f"checkpoint-{frame_count}",
        }
    )
    report_path.write_text(json.dumps(report), encoding="utf-8")


def test_compare_runs_pairs_by_sample_and_resamples_scene_groups(tmp_path):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    reference_meta = tmp_path / "reference-training.json"
    candidate_meta = tmp_path / "candidate-training.json"
    reference_corpus = tmp_path / "reference-corpus.json"
    candidate_corpus = tmp_path / "candidate-corpus.json"
    _write_prediction_run(reference_dir, 8, 0.0)
    _write_prediction_run(candidate_dir, 16, 0.02)
    _write_training_metadata(reference_meta, 8)
    _write_training_metadata(candidate_meta, 16)
    _write_corpus_report(reference_corpus, 8)
    _write_corpus_report(candidate_corpus, 16)
    _bind_evaluation_receipts(reference_dir, reference_meta, reference_corpus, 8)
    _bind_evaluation_receipts(candidate_dir, candidate_meta, candidate_corpus, 16)

    result = compare_runs(
        reference_evaluation_dir=reference_dir,
        candidate_evaluation_dir=candidate_dir,
        reference_training_metadata=reference_meta,
        candidate_training_metadata=candidate_meta,
        reference_corpus_report=reference_corpus,
        candidate_corpus_report=candidate_corpus,
        replicates=200,
        seed=17,
    )

    assert result["reference_frame_count"] == 8
    assert result["candidate_frame_count"] == 16
    assert result["n_videos"] == 12
    assert result["scene_groups"] == 3
    assert result["paired_scene_group_bootstrap_candidate_minus_reference"][
        "valid_replicates"
    ] == 200
    assert result["sealed_test_accessed"] is False


def test_compare_runs_rejects_different_dev_sample_sets(tmp_path):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    reference_meta = tmp_path / "reference-training.json"
    candidate_meta = tmp_path / "candidate-training.json"
    reference_corpus = tmp_path / "reference-corpus.json"
    candidate_corpus = tmp_path / "candidate-corpus.json"
    _write_prediction_run(reference_dir, 8, 0.0)
    _write_prediction_run(candidate_dir, 16, 0.02, alter_first_sample=True)
    _write_training_metadata(reference_meta, 8)
    _write_training_metadata(candidate_meta, 16)
    _write_corpus_report(reference_corpus, 8)
    _write_corpus_report(candidate_corpus, 16)
    _bind_evaluation_receipts(reference_dir, reference_meta, reference_corpus, 8)
    _bind_evaluation_receipts(candidate_dir, candidate_meta, candidate_corpus, 16)

    with pytest.raises(ValueError, match="sample IDs differ"):
        compare_runs(
            reference_evaluation_dir=reference_dir,
            candidate_evaluation_dir=candidate_dir,
            reference_training_metadata=reference_meta,
            candidate_training_metadata=candidate_meta,
            reference_corpus_report=reference_corpus,
            candidate_corpus_report=candidate_corpus,
            replicates=200,
        )
