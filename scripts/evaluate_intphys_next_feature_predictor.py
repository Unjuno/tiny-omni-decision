"""Run one fixed IntPhys dev diagnostic against a causal LSTM and persistence baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.audit_videoprism_checkpoint import CHECKPOINT_SHA256, MODEL_REVISION
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
from tiny_omni_decision.intphys_metrics import (
    grouped_paired_bootstrap,
    threshold_metrics,
    validation_tail_threshold,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dev_features(
    path: Path,
) -> tuple[np.ndarray, list[str], list[str], list[str], np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "features",
            "sample_ids",
            "group_ids",
            "blocks",
            "labels",
            "normalized_media_sha256",
        }
        if not required.issubset(payload.files):
            raise ValueError(
                f"dev feature archive missing arrays: {sorted(required - set(payload.files))}"
            )
        features = np.asarray(payload["features"], dtype=np.float32)
        sample_ids = payload["sample_ids"].astype(str).tolist()
        group_ids = payload["group_ids"].astype(str).tolist()
        blocks = payload["blocks"].astype(str).tolist()
        labels = np.asarray(payload["labels"], dtype=np.int8)
        media_hashes = payload["normalized_media_sha256"].astype(str).tolist()
    count = len(features) if features.ndim else 0
    if (
        features.ndim != 3
        or features.shape[0] == 0
        or features.shape[1] < 2
        or features.shape[2] < 1
        or not np.isfinite(features).all()
        or any(
            len(items) != count for items in (sample_ids, group_ids, blocks, labels, media_hashes)
        )
        or len(set(sample_ids)) != count
        or not np.isin(labels, [0, 1]).all()
        or any(not value for values in (sample_ids, group_ids, blocks) for value in values)
        or any(not value for value in media_hashes)
    ):
        raise ValueError("invalid finite dev arrays or misaligned sample metadata")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        groups[group_id].append(index)
    for group_id, indices in groups.items():
        if len(indices) != 4 or int(labels[indices].sum()) != 2:
            raise ValueError(f"dev group {group_id} is not a four-video 2/2 label group")
    return features, sample_ids, group_ids, blocks, labels, media_hashes


def load_split_identities(path: Path) -> tuple[list[str], list[str], list[str]]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"sample_ids", "group_ids", "sampled_rgb_pixels_sha256"}
        if not required.issubset(payload.files):
            missing = sorted(required - set(payload.files))
            raise ValueError(
                f"feature cache lacks split identity arrays: {missing}"
            )
        sample_ids = payload["sample_ids"].astype(str).tolist()
        group_ids = payload["group_ids"].astype(str).tolist()
        media_hashes = payload["sampled_rgb_pixels_sha256"].astype(str).tolist()
    if (
        not sample_ids
        or any(len(values) != len(sample_ids) for values in (group_ids, media_hashes))
        or len(set(sample_ids)) != len(sample_ids)
        or any(not value for values in (sample_ids, group_ids, media_hashes) for value in values)
    ):
        raise ValueError("invalid split IDs, groups, or sampled-media identities")
    return sample_ids, group_ids, media_hashes


def validate_cross_split_identities(
    splits: dict[str, tuple[list[str], list[str], list[str]]],
) -> dict[str, int]:
    """Reject sample, underlying group, or normalized sampled-media overlap."""
    if set(splits) != {"train", "validation", "dev"}:
        raise ValueError("exact train, validation, and dev identity lists are required")
    overlaps = {"sample_id": 0, "group_id": 0, "sampled_media_sha256": 0}
    names = ["train", "validation", "dev"]
    for left_index, left_name in enumerate(names):
        left_ids, left_groups, left_media = splits[left_name]
        for right_name in names[left_index + 1 :]:
            right_ids, right_groups, right_media = splits[right_name]
            identity_pairs = (
                ("sample_id", left_ids, right_ids),
                ("group_id", left_groups, right_groups),
                ("sampled_media_sha256", left_media, right_media),
            )
            for identity_name, left_values, right_values in identity_pairs:
                overlap = set(left_values) & set(right_values)
                overlaps[identity_name] += len(overlap)
                if overlap:
                    raise ValueError(
                        f"{identity_name} overlap between {left_name} and {right_name}: "
                        f"{len(overlap)}"
                    )
    return overlaps


def anomaly_scores(
    model: Any, features: np.ndarray, *, batch_size: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    from tiny_omni_decision.temporal_prediction import per_sequence_cosine_error

    if batch_size < 1:
        raise ValueError("batch size must be positive")
    model.eval()
    learned: list[np.ndarray] = []
    persistence: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(features[start : start + batch_size]))
            learned.append(per_sequence_cosine_error(model(batch), batch).cpu().numpy())
            persistence.append(per_sequence_cosine_error(batch[:, :-1], batch).cpu().numpy())
    return np.concatenate(learned), np.concatenate(persistence)


def validate_receipts(
    *,
    checkpoint: Path,
    train_metadata_path: Path,
    train_corpus_report_path: Path,
    train_features_path: Path,
    validation_features_path: Path,
    dev_features_path: Path,
    dev_report_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train_metadata = json.loads(train_metadata_path.read_text(encoding="utf-8"))
    train_corpus_report = json.loads(train_corpus_report_path.read_text(encoding="utf-8"))
    dev_report = json.loads(dev_report_path.read_text(encoding="utf-8"))
    if train_metadata.get("status") != "complete_train_only_next_feature_predictor":
        raise ValueError("training receipt is not a completed train-only Path A run")
    if train_corpus_report.get("all_train_labels_possible") is not True:
        raise ValueError("Path A requires the audited possible-only training corpus")
    if train_corpus_report.get("status") != "complete_intphys_train_frozen_feature_extraction":
        raise ValueError("training corpus receipt is not a completed pinned extraction")
    if (
        train_corpus_report.get("archive_bytes") != EXPECTED_TRAIN_BYTES
        or train_corpus_report.get("archive_md5") != EXPECTED_TRAIN_MD5
        or train_corpus_report.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or train_corpus_report.get("model_revision") != MODEL_REVISION
        or train_corpus_report.get("videoprism_source_commit") != VIDEOPRISM_SOURCE_COMMIT
    ):
        raise ValueError(
            "training corpus source or frozen encoder does not match pinned R1 receipts"
        )
    corpus_feature_shape = train_corpus_report.get("feature_shape", [])
    dev_feature_shape = dev_report.get("feature_shape", [])
    train_frame_count = train_corpus_report.get(
        "frame_count", corpus_feature_shape[0] if corpus_feature_shape else None
    )
    dev_frame_count = dev_report.get(
        "frame_count", dev_feature_shape[0] if dev_feature_shape else None
    )
    metadata_frame_count = train_metadata.get("sequence_length", train_frame_count)
    if (
        not isinstance(train_frame_count, int)
        or train_frame_count < 2
        or dev_frame_count != train_frame_count
        or metadata_frame_count != train_frame_count
    ):
        raise ValueError("train, validation, and dev frame counts do not match")
    if train_metadata.get("best_checkpoint_sha256") != sha256_file(checkpoint):
        raise ValueError("predictor checkpoint hash does not match training receipt")
    if train_metadata.get("train_features_sha256") != sha256_file(train_features_path):
        raise ValueError("train feature hash does not match training receipt")
    if train_metadata.get("validation_features_sha256") != sha256_file(validation_features_path):
        raise ValueError("internal validation feature hash does not match training receipt")
    recorded_feature_hashes = train_corpus_report.get("feature_archive_sha256", {})
    if recorded_feature_hashes.get("train") != sha256_file(train_features_path):
        raise ValueError("train feature hash does not match the pinned corpus receipt")
    if recorded_feature_hashes.get("validation") != sha256_file(validation_features_path):
        raise ValueError("validation feature hash does not match the pinned corpus receipt")
    if dev_report.get("status") != "complete_intphys_dev_frozen_features_for_one_shot_diagnostic":
        raise ValueError("dev feature receipt has unexpected status")
    if (
        dev_report.get("archive_bytes") != EXPECTED_DEV_BYTES
        or dev_report.get("archive_md5") != EXPECTED_DEV_MD5
        or dev_report.get("archive_sha256") != EXPECTED_DEV_SHA256
        or dev_report.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or dev_report.get("model_revision") != MODEL_REVISION
        or dev_report.get("videoprism_source_commit") != VIDEOPRISM_SOURCE_COMMIT
    ):
        raise ValueError("dev source or frozen encoder does not match pinned R1 receipts")
    if dev_report.get("feature_file_sha256") != sha256_file(dev_features_path):
        raise ValueError("dev feature hash does not match extraction receipt")
    if dev_report.get("sealed_test_accessed") is not False:
        raise ValueError("sealed test access must be explicitly false")
    return train_metadata, dev_report


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-metadata", type=Path, required=True)
    parser.add_argument("--training-corpus-report", type=Path, required=True)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--validation-features", type=Path, required=True)
    parser.add_argument("--dev-features", type=Path, required=True)
    parser.add_argument("--dev-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    return parser.parse_args()


def main() -> None:
    import torch

    from tiny_omni_decision.temporal_prediction import NextFrameFeaturePredictor

    args = _args()
    checkpoint = args.checkpoint.resolve(strict=True)
    training_metadata_path = args.training_metadata.resolve(strict=True)
    train_corpus_report_path = args.training_corpus_report.resolve(strict=True)
    train_path = args.train_features.resolve(strict=True)
    validation_path = args.validation_features.resolve(strict=True)
    dev_path = args.dev_features.resolve(strict=True)
    dev_report_path = args.dev_report.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {output_dir}")
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    training_metadata, dev_report = validate_receipts(
        checkpoint=checkpoint,
        train_metadata_path=training_metadata_path,
        train_corpus_report_path=train_corpus_report_path,
        train_features_path=train_path,
        validation_features_path=validation_path,
        dev_features_path=dev_path,
        dev_report_path=dev_report_path,
    )
    with np.load(validation_path, allow_pickle=False) as payload:
        validation_features = np.asarray(payload["features"], dtype=np.float32)
    train_ids, train_groups, train_media_hashes = load_split_identities(train_path)
    validation_ids, validation_groups, validation_media_hashes = load_split_identities(
        validation_path
    )
    if (
        validation_features.ndim != 3
        or len(validation_features) < 2
        or not np.isfinite(validation_features).all()
        or len(validation_ids) != len(validation_features)
        or len(validation_groups) != len(validation_features)
    ):
        raise ValueError("invalid internal validation features or identities")
    dev_features, sample_ids, group_ids, blocks, labels, dev_media_hashes = load_dev_features(
        dev_path
    )
    identity_overlap = validate_cross_split_identities(
        {
            "train": (train_ids, train_groups, train_media_hashes),
            "validation": (validation_ids, validation_groups, validation_media_hashes),
            "dev": (sample_ids, group_ids, dev_media_hashes),
        }
    )
    if dev_features.shape[1:] != validation_features.shape[1:]:
        raise ValueError("dev and internal validation feature dimensions differ")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    width = int(payload["feature_width"])
    hidden = int(payload["hidden_width"])
    if width != validation_features.shape[-1] or int(payload["best_epoch"]) != int(
        training_metadata["selected_epoch"]
    ):
        raise ValueError("checkpoint architecture or selected epoch disagrees with receipt")
    model = NextFrameFeaturePredictor(feature_width=width, hidden_width=hidden)
    model.load_state_dict(payload["state_dict"], strict=True)
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise RuntimeError("Path A evaluation must run on CPU")

    validation_learned, validation_persistence = anomaly_scores(
        model, validation_features, batch_size=args.batch_size
    )
    dev_learned, dev_persistence = anomaly_scores(model, dev_features, batch_size=args.batch_size)
    learned_threshold = validation_tail_threshold(validation_learned)
    persistence_threshold = validation_tail_threshold(validation_persistence)
    learned_metrics = threshold_metrics(labels, dev_learned, learned_threshold)
    persistence_metrics = threshold_metrics(labels, dev_persistence, persistence_threshold)
    bootstrap = grouped_paired_bootstrap(
        labels,
        dev_learned,
        dev_persistence,
        group_ids,
        first_threshold=learned_threshold,
        second_threshold=persistence_threshold,
        replicates=args.bootstrap_replicates,
        seed=17,
    )
    by_block: dict[str, dict[str, Any]] = {}
    for block in sorted(set(blocks)):
        indices = np.asarray([index for index, value in enumerate(blocks) if value == block])
        by_block[block] = {
            "learned_lstm": threshold_metrics(
                labels[indices], dev_learned[indices], learned_threshold
            ),
            "persistence": threshold_metrics(
                labels[indices], dev_persistence[indices], persistence_threshold
            ),
        }
    rows = [
        {
            "sample_id": sample_ids[index],
            "scene_group_id": group_ids[index],
            "block": blocks[index],
            "label_impossible": int(labels[index]),
            "learned_lstm_anomaly_score": float(dev_learned[index]),
            "learned_lstm_predicted_impossible": bool(dev_learned[index] > learned_threshold),
            "persistence_anomaly_score": float(dev_persistence[index]),
            "persistence_predicted_impossible": bool(
                dev_persistence[index] > persistence_threshold
            ),
        }
        for index in range(len(sample_ids))
    ]
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = output_dir / "dev-predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    report = {
        "status": "complete_single_fixed_intphys_dev_diagnostic",
        "started_at_utc": started_at,
        "ended_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "device": "CPU",
        "command": sys.argv,
        "scope": "IntPhys dev possible-vs-impossible temporal feature anomaly diagnostic only",
        "checkpoint_sha256": sha256_file(checkpoint),
        "training_metadata_sha256": sha256_file(training_metadata_path),
        "training_corpus_report_sha256": sha256_file(train_corpus_report_path),
        "train_features_sha256": sha256_file(train_path),
        "validation_features_sha256": sha256_file(validation_path),
        "dev_features_sha256": sha256_file(dev_path),
        "dev_report_sha256": sha256_file(dev_report_path),
        "dev_scene_groups": len(set(group_ids)),
        "dev_video_count": len(sample_ids),
        "cross_split_identity_overlap": identity_overlap,
        "dev_prediction_rows_sha256": sha256_file(predictions_path),
        "threshold_rule": (
            "95th percentile (higher order statistic) of internal possible-only "
            "validation anomaly scores"
        ),
        "model_selection_used_dev_labels": False,
        "dev_used_for_fit_or_calibration": False,
        "sealed_test_accessed": False,
        "scores_are_calibrated_class_probabilities": False,
        "probability_metrics": (
            "NLL/Brier/ECE omitted; anomaly scores are not calibrated class probabilities"
        ),
        "learned_lstm": learned_metrics,
        "persistence_baseline": persistence_metrics,
        "by_intphys_block": by_block,
        "macro_block_roc_auc": {
            "learned_lstm": float(
                np.mean([value["learned_lstm"]["roc_auc"] for value in by_block.values()])
            ),
            "persistence": float(
                np.mean([value["persistence"]["roc_auc"] for value in by_block.values()])
            ),
        },
        "paired_scene_group_bootstrap": bootstrap,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": 17,
        "limitations": [
            "The dev partition is not a blind sealed audit and was previously inspected "
            "for source integrity.",
            "A one-class normality threshold is not a calibrated posterior probability.",
            "Results describe only this IntPhys task and frozen VideoPrism representation.",
        ],
    }
    report_path = output_dir / "evaluation-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
