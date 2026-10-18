"""Train matched frozen-feature CLEVRER video Decision readouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.nn import functional as F

from tiny_omni_decision.classification_json import classification_json
from tiny_omni_decision.textfree_video import (
    EncodedVideoRows,
    TextFreeVideoReadout,
    VideoVocab,
    build_video_vocab,
    encode_video_rows,
    validate_video_splits,
)

SEED = 17
EPOCHS = 256
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
DATASET_REVISION = "98b842082ba4f7c18b6b9e3f39145871782a65ef"
VIDEO_REVISION = "be719a406d563b66f0ac969e7c94bab8e997c81a"
VIDEO_SOURCE_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
VIDEO_CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
VIDEO_ENCODER_PARAMETER_COUNT = 114_365_184
TAXONOMIES = ("count", "exist", "query_color", "query_material", "query_shape")
QUESTION_TYPES = ("temporal_descriptive", "static_descriptive")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pretrained_reuse/textfree_video_clevrer_v0.json"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _normalized_content(row: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    question = " ".join(str(row["question"]).casefold().split())
    options = tuple(sorted(" ".join(str(value).casefold().split()) for value in row["options"]))
    return question, options


def _ordered_ids_sha256(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256("\n".join(str(row["id"]) for row in rows).encode()).hexdigest()


def _training_schedule(
    *, train_count: int, batch_size: int, epochs: int, max_optimizer_updates: int
) -> dict[str, int]:
    if train_count < 1 or batch_size < 1 or epochs < 1 or max_optimizer_updates < 1:
        raise ValueError("training schedule values must be positive")
    updates_per_epoch = math.ceil(train_count / batch_size)
    total_updates = epochs * updates_per_epoch
    if total_updates != max_optimizer_updates:
        raise ValueError(
            "configured epochs and batch size do not match max_optimizer_updates: "
            f"{epochs} * {updates_per_epoch} != {max_optimizer_updates}"
        )
    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "updates_per_epoch": updates_per_epoch,
        "total_optimizer_updates": total_updates,
        "full_run_example_exposures": epochs * train_count,
    }


def _validation_interval_updates(
    *, train_count: int, batch_size: int, validation_interval_epochs: int, epochs: int
) -> int:
    if validation_interval_epochs < 1 or epochs < 1 or epochs % validation_interval_epochs:
        raise ValueError("validation interval must be positive and divide the training epochs")
    return math.ceil(train_count / batch_size) * validation_interval_epochs


def _source_rows_gate(
    rows: list[dict[str, Any]], sample_dir: Path, split: str, revision: str
) -> dict[str, Any]:
    if not rows or any(row.get("split") != split for row in rows):
        raise ValueError(f"empty or incorrectly marked {split} rows")
    scene_media: dict[str, str] = {}
    verified_media: dict[str, str] = {}
    for row in rows:
        if row.get("source") != f"MIT-IBM/CLEVRER@{revision}":
            raise ValueError(f"unexpected source revision in {row.get('id')}")
        if row.get("question_type") not in QUESTION_TYPES:
            raise ValueError(f"unsupported CLEVRER descriptive task: {row.get('question_type')}")
        media_hash = str(row["media_sha256"])
        scene = str(row["scene_group_id"])
        if scene in scene_media and scene_media[scene] != media_hash:
            raise ValueError(f"scene group maps to multiple assets: {scene}")
        scene_media[scene] = media_hash
        relative = Path(str(row["media_path"]))
        media_path = (sample_dir / relative).resolve(strict=True)
        try:
            media_path.relative_to(sample_dir)
        except ValueError as error:
            raise ValueError(f"media path escapes frozen sample directory: {media_path}") from error
        if media_hash not in verified_media:
            actual_hash = _sha256(media_path)
            if actual_hash != media_hash:
                raise ValueError(f"media hash mismatch for {row['id']}")
            verified_media[media_hash] = actual_hash
    return {
        "rows": len(rows),
        "unique_sample_ids": len({str(row["id"]) for row in rows}),
        "unique_scenes": len(scene_media),
        "unique_media_assets": len(verified_media),
        "question_type_counts": dict(
            sorted(Counter(str(row["question_type"]) for row in rows).items())
        ),
        "taxonomy_counts": dict(sorted(Counter(str(row["taxonomy"]) for row in rows).items())),
        "ordered_sample_id_sha256": _ordered_ids_sha256(rows),
        "verified_media_sha256": sorted(verified_media),
    }


def _validate_inputs(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    source = config["source"]
    feature_config = config["features"]
    expected = config.get("expected", {})
    integrity = config.get("integrity", {})
    sample_dir = Path(source["sample_dir"]).resolve(strict=True)
    train_path, validation_path = sample_dir / "train.jsonl", sample_dir / "validation.jsonl"
    fetch_path = sample_dir / "fetch-report.json"
    gate_path = Path(
        source.get("integrity_report_path", sample_dir / "pretraining-gate.json")
    ).resolve(strict=True)
    expected_hashes = {
        train_path: source["train_manifest_sha256"],
        validation_path: source["validation_manifest_sha256"],
        fetch_path: source["fetch_report_sha256"],
        gate_path: source.get("integrity_report_sha256", source.get("pretraining_gate_sha256")),
    }
    for path, expected_hash in expected_hashes.items():
        actual = _sha256(path)
        if actual != expected_hash:
            raise ValueError(f"pinned input hash mismatch: {path}")
    if source["revision"] != DATASET_REVISION or source["license"] != "CC0-1.0":
        raise ValueError("unexpected data revision or license")
    repository_root = Path(__file__).resolve().parents[1]
    source_manifest_path = repository_root / "manifests" / "candidates" / "clevrer.yaml"
    source_manifest = yaml.safe_load(source_manifest_path.read_text(encoding="utf-8"))
    if (
        source_manifest.get("revision") != DATASET_REVISION
        or source_manifest.get("license") != "CC0-1.0"
        or source_manifest.get("derivative_model_training_allowed") is not True
        or source_manifest.get("usage") != "training"
        or source_manifest.get("trust_status") != "trusted"
    ):
        raise ValueError("repository source manifest does not admit this CLEVRER training use")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") == "pretraining_gate_pass":
        if (
            gate.get("media_sha_overlap") != 0
            or gate.get("normalized_question_overlap") != 0
            or gate.get("archive_sha256_verified") is not False
        ):
            raise ValueError("frozen CLEVRER pretraining gate does not match the recorded policy")
    elif gate.get("status") == "passed" and "group_integrity" in gate:
        groups = gate["group_integrity"]
        if (
            gate.get("all_media_sha256_verified") is not True
            or groups.get("train_validation_scene_overlap") != 0
            or groups.get("train_validation_media_overlap") != 0
            or gate.get("sample_hashes", {}).get("train_jsonl")
            != source["train_manifest_sha256"]
            or gate.get("sample_hashes", {}).get("validation_jsonl")
            != source["validation_manifest_sha256"]
        ):
            raise ValueError(
                "CLEVRER media/split integrity receipt does not match the frozen sample"
            )
    else:
        raise ValueError("unsupported CLEVRER sample integrity receipt")

    train_rows, validation_rows = _read_rows(train_path), _read_rows(validation_path)
    validate_video_splits(
        train_rows,
        validation_rows,
        allow_normalized_content_overlap=bool(
            integrity.get("allow_normalized_question_option_overlap", False)
        ),
    )
    if len(train_rows) != expected.get("train_rows", 512) or len(validation_rows) != expected.get(
        "validation_rows", 400
    ):
        raise ValueError("unexpected frozen CLEVRER sample cardinality")
    train_report = _source_rows_gate(train_rows, sample_dir, "train", DATASET_REVISION)
    validation_report = _source_rows_gate(
        validation_rows, sample_dir, "validation", DATASET_REVISION
    )
    content_overlap = {_normalized_content(row) for row in train_rows} & {
        _normalized_content(row) for row in validation_rows
    }
    if content_overlap and not integrity.get("allow_normalized_question_option_overlap", False):
        raise ValueError("normalized question/options overlap remains across splits")
    if set(train_report["question_type_counts"]) != set(QUESTION_TYPES) or set(
        validation_report["question_type_counts"]
    ) != set(QUESTION_TYPES):
        raise ValueError("both static and temporal descriptive CLEVRER tasks must be present")

    feature_path = Path(feature_config["cache"]).resolve(strict=True)
    feature_report_path = Path(feature_config["cache_report"]).resolve(strict=True)
    feature_report = json.loads(feature_report_path.read_text(encoding="utf-8"))
    if feature_report.get("status") != "complete_videoprism_framewise_feature_cache":
        raise ValueError("framewise feature cache is incomplete")
    if (
        feature_report.get("dataset_revision") != f"MIT-IBM/CLEVRER@{DATASET_REVISION}"
        or feature_report.get("model_revision") != VIDEO_REVISION
        or feature_report.get("upstream_source_commit") != VIDEO_SOURCE_COMMIT
        or feature_report.get("checkpoint_sha256") != VIDEO_CHECKPOINT_SHA256
        or feature_config.get("encoder_parameter_count") != VIDEO_ENCODER_PARAMETER_COUNT
        or feature_report.get("frame_count") != 8
        or feature_report.get("feature_dimension") != 768
    ):
        raise ValueError("framewise feature provenance does not match the frozen config")
    if (
        feature_report.get("train_manifest_sha256", {}).get("train.jsonl")
        != feature_config.get("source_train_manifest_sha256", source["train_manifest_sha256"])
        or feature_report.get("evaluation_manifest_sha256", {}).get("validation.jsonl")
        != feature_config.get(
            "source_validation_manifest_sha256", source["validation_manifest_sha256"]
        )
    ):
        raise ValueError("feature cache was not extracted from the pinned sample manifests")
    if feature_report.get("feature_cache_sha256") != _sha256(feature_path):
        raise ValueError("framewise feature cache hash mismatch")
    with np.load(feature_path, allow_pickle=False) as archive:
        media_hashes = archive["media_sha256"].astype(str).tolist()
        values = archive["features"]
        expected_feature_count = int(expected.get("feature_assets", 456))
        if values.shape != (expected_feature_count, 8, 768) or values.dtype != np.float32:
            raise ValueError(
                f"expected [{expected_feature_count},8,768] float32 framewise features"
            )
        if not np.isfinite(values).all() or len(media_hashes) != len(set(media_hashes)):
            raise ValueError("feature cache contains non-finite values or duplicate media")
        features_by_media = {
            media_hash: values[index].copy() for index, media_hash in enumerate(media_hashes)
        }
    train_media = set(train_report["verified_media_sha256"])
    validation_media = set(validation_report["verified_media_sha256"])
    if not train_media.issubset(features_by_media) or not validation_media.issubset(
        features_by_media
    ):
        raise ValueError("framewise cache does not cover every pinned train/validation asset")
    feature_train_media = set(feature_report.get("train_media_hashes", []))
    feature_validation_media = set(feature_report.get("evaluation_media_hashes", []))
    if not train_media.issubset(feature_train_media) or (
        feature_validation_media != validation_media
    ):
        raise ValueError("feature report media identity list does not match the sample manifests")

    vocab = build_video_vocab(train_rows)
    train_encoded = encode_video_rows(train_rows, features_by_media, vocab)
    validation_encoded = encode_video_rows(validation_rows, features_by_media, vocab)
    program_unknown = sum(
        token not in vocab.program_to_id for row in validation_rows for token in row["program"]
    )
    option_unknown = sum(
        option not in vocab.option_to_id for row in validation_rows for option in row["options"]
    )
    if program_unknown or option_unknown:
        raise ValueError("validation contains program or option tokens not seen in training")
    return {
        "config_sha256": _sha256(config_path),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "sample_dir": str(sample_dir),
        "train_jsonl_sha256": _sha256(train_path),
        "validation_jsonl_sha256": _sha256(validation_path),
        "fetch_report_sha256": _sha256(fetch_path),
        "pretraining_gate_sha256": _sha256(gate_path),
        "feature_cache": str(feature_path),
        "feature_cache_sha256": _sha256(feature_path),
        "feature_cache_report_sha256": _sha256(feature_report_path),
        "feature_extraction_seconds": float(feature_report["extraction_seconds"]),
        "feature_extraction_devices": feature_report.get("devices", []),
        "feature_extraction_peak_working_set_bytes": feature_report.get("peak_working_set_bytes"),
        "feature_unique_video_count": int(feature_report["unique_video_count"]),
        "train": train_report,
        "validation": validation_report,
        "sample_id_overlap": 0,
        "scene_group_overlap": 0,
        "media_identity_overlap": 0,
        "normalized_question_option_overlap": len(content_overlap),
        "validation_program_unknown_token_count": program_unknown,
        "validation_option_unknown_token_count": option_unknown,
        "program_vocabulary_size": len(vocab.program_to_id),
        "taxonomy_vocabulary": sorted(vocab.taxonomy_to_id),
        "option_vocabulary": sorted(vocab.option_to_id),
        "train_examples": len(train_encoded.sample_ids),
        "validation_examples": len(validation_encoded.sample_ids),
        "unique_train_assets": len(train_media),
        "unique_validation_assets": len(validation_media),
        "archive_full_sha256_verified": False,
        "sealed_audit_loaded": False,
    }


def _choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(requested)
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        if free < 512 * 1024 * 1024:
            raise RuntimeError(f"less than 512 MiB free before training ({free} bytes)")
        _reset_peak_memory_stats(device)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return device


def _reset_peak_memory_stats(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _move(batch: EncodedVideoRows, device: torch.device) -> EncodedVideoRows:
    return EncodedVideoRows(
        video=batch.video.to(device),
        program_ids=batch.program_ids.to(device),
        taxonomy_ids=batch.taxonomy_ids.to(device),
        option_ids=batch.option_ids.to(device),
        option_mask=batch.option_mask.to(device),
        targets=batch.targets.to(device),
        sample_ids=batch.sample_ids,
    )


def _subset(batch: EncodedVideoRows, indexes: Tensor) -> EncodedVideoRows:
    return EncodedVideoRows(
        video=batch.video.index_select(0, indexes),
        program_ids=batch.program_ids.index_select(0, indexes),
        taxonomy_ids=batch.taxonomy_ids.index_select(0, indexes),
        option_ids=batch.option_ids.index_select(0, indexes),
        option_mask=batch.option_mask.index_select(0, indexes),
        targets=batch.targets.index_select(0, indexes),
        sample_ids=(),
    )


def _logits(model: TextFreeVideoReadout, batch: EncodedVideoRows) -> Tensor:
    return model(
        batch.video, batch.program_ids, batch.taxonomy_ids, batch.option_ids, batch.option_mask
    )


def _metric_tensors(logits: Tensor, targets: Tensor, option_mask: Tensor) -> dict[str, float | int]:
    with torch.inference_mode():
        probabilities = torch.softmax(logits.float(), dim=-1)
        probabilities = probabilities * option_mask
        confidence, predictions = probabilities.max(dim=-1)
        correct = predictions.eq(targets).float()
        one_hot = F.one_hot(targets, num_classes=logits.shape[-1]).float() * option_mask
        nll = F.cross_entropy(logits.float(), targets, reduction="none")
        brier = (probabilities - one_hot).square().sum(dim=-1)
        ece = torch.zeros((), dtype=torch.float32, device=logits.device)
        for bin_index in range(15):
            lower = bin_index / 15
            mask = (confidence >= lower) & (
                confidence < (bin_index + 1) / 15 if bin_index < 14 else confidence <= 1
            )
            if bool(mask.any()):
                ece += mask.float().mean() * (correct[mask].mean() - confidence[mask].mean()).abs()
        return {
            "count": int(len(targets)),
            "accuracy": float(correct.mean().item()),
            "nll": float(nll.mean().item()),
            "brier": float(brier.mean().item()),
            "ece_15_bins": float(ece.item()),
            "mean_confidence": float(confidence.mean().item()),
        }


def _evaluate(
    model: TextFreeVideoReadout, batch: EncodedVideoRows, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    model.eval()
    with torch.inference_mode():
        all_logits = _logits(model, batch).float()
    all_indexes = torch.arange(len(rows), device=all_logits.device)
    overall = _metric_tensors(all_logits, batch.targets, batch.option_mask)
    by_question_type: dict[str, Any] = {}
    by_taxonomy: dict[str, Any] = {}
    by_group: dict[str, Any] = {}
    group_nll_by_type: dict[str, list[float]] = defaultdict(list)
    for question_type in QUESTION_TYPES:
        indexes = [index for index, row in enumerate(rows) if row["question_type"] == question_type]
        selected = torch.tensor(indexes, dtype=torch.long, device=all_logits.device)
        by_question_type[question_type] = _metric_tensors(
            all_logits.index_select(0, selected),
            batch.targets.index_select(0, selected),
            batch.option_mask.index_select(0, selected),
        )
        for taxonomy in TAXONOMIES:
            group_indexes = [
                index
                for index, row in enumerate(rows)
                if row["question_type"] == question_type and row["taxonomy"] == taxonomy
            ]
            if not group_indexes:
                continue
            group_tensor = torch.tensor(group_indexes, dtype=torch.long, device=all_logits.device)
            group_metrics = _metric_tensors(
                all_logits.index_select(0, group_tensor),
                batch.targets.index_select(0, group_tensor),
                batch.option_mask.index_select(0, group_tensor),
            )
            by_group[f"{question_type}/{taxonomy}"] = group_metrics
            by_taxonomy.setdefault(taxonomy, []).append(group_metrics)
            group_nll_by_type[question_type].append(float(group_metrics["nll"]))
    selector_by_type = {
        question_type: float(np.mean(group_nll_by_type[question_type]))
        for question_type in QUESTION_TYPES
    }
    selector = float(np.mean(list(selector_by_type.values())))
    macro_by_type = {
        metric: float(np.mean([by_question_type[name][metric] for name in QUESTION_TYPES]))
        for metric in ("accuracy", "nll", "brier", "ece_15_bins")
    }
    macro_by_taxonomy = {
        taxonomy: {
            metric: float(np.mean([record[metric] for record in records]))
            for metric in ("accuracy", "nll", "brier", "ece_15_bins")
        }
        for taxonomy, records in by_taxonomy.items()
    }
    return {
        "overall_example_weighted": overall,
        "macro_question_type": macro_by_type,
        "macro_taxonomy": macro_by_taxonomy,
        "by_question_type": by_question_type,
        "by_question_type_and_taxonomy": by_group,
        "selector_score_macro_nll": selector,
        "selector_nll_by_question_type": selector_by_type,
        "logits": all_logits,
        "row_indexes": all_indexes,
    }


def _write_predictions(
    path: Path,
    logits: Tensor,
    batch: EncodedVideoRows,
    rows: list[dict[str, Any]],
    option_to_id: dict[str, int],
) -> None:
    probabilities = torch.softmax(logits.float(), dim=-1).cpu()
    option_ids = batch.option_ids.cpu()
    option_mask = batch.option_mask.cpu()
    targets = batch.targets.cpu()
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(rows):
            choices = []
            count = int(option_mask[index].sum())
            ids = option_ids[index, :count].tolist()
            reverse = {value: key for key, value in option_to_id.items()}
            values = [reverse.get(int(value), "<unk>") for value in ids]
            option_mapping = [{"option_id": class_id, "class_id": class_id} for class_id in values]
            decision = classification_json(
                probabilities[index, :count].tolist(), values, option_mapping
            )
            pred_index = values.index(str(decision["selected_option_id"]))
            target_index = int(targets[index])
            for class_id, probability in zip(
                values, probabilities[index, :count].tolist(), strict=True
            ):
                choices.append(
                    {
                        "option_id": class_id,
                        "class_id": class_id,
                        "probability": probability,
                    }
                )
            stream.write(
                json.dumps(
                    {
                        "id": str(row["id"]),
                        "scene_group_id": str(row["scene_group_id"]),
                        "media_sha256": str(row["media_sha256"]),
                        "question_type": str(row["question_type"]),
                        "taxonomy": str(row["taxonomy"]),
                        "target": values[target_index],
                        "prediction": values[pred_index],
                        "selected_class_id": decision["selected_option_id"],
                        **decision,
                        "options": choices,
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )


def _model_prediction(model: TextFreeVideoReadout, batch: EncodedVideoRows) -> Tensor:
    model.eval()
    with torch.inference_mode():
        return _logits(model, batch).float()


def _with_video(batch: EncodedVideoRows, video: Tensor) -> EncodedVideoRows:
    return EncodedVideoRows(
        video=video,
        program_ids=batch.program_ids,
        taxonomy_ids=batch.taxonomy_ids,
        option_ids=batch.option_ids,
        option_mask=batch.option_mask,
        targets=batch.targets,
        sample_ids=batch.sample_ids,
    )


def _scene_shuffled_video(batch: EncodedVideoRows, rows: list[dict[str, Any]]) -> Tensor:
    scene_features: dict[str, Tensor] = {}
    row_scenes: list[str] = []
    for index, row in enumerate(rows):
        scene = str(row["scene_group_id"])
        feature = batch.video[index]
        if scene in scene_features and not torch.equal(scene_features[scene], feature):
            raise ValueError(f"multiple feature sequences for scene group {scene}")
        scene_features[scene] = feature
        row_scenes.append(scene)
    scenes = np.asarray(sorted(scene_features), dtype=str)
    if len(scenes) < 2:
        raise ValueError("scene-shuffle control requires at least two scenes")
    order = np.random.default_rng(SEED).permutation(scenes)
    source_by_target = {
        str(target): str(order[(index - 1) % len(order)]) for index, target in enumerate(order)
    }
    if any(target == source for target, source in source_by_target.items()):
        raise RuntimeError("scene shuffle unexpectedly retained a matched scene")
    return torch.stack([scene_features[source_by_target[scene]] for scene in row_scenes])


def _load_row_metrics(path: Path) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        options = row["options"]
        target_indexes = [
            index for index, option in enumerate(options) if option["class_id"] == row["target"]
        ]
        if len(target_indexes) != 1:
            raise ValueError(f"prediction target does not match one option: {row['id']}")
        probabilities = np.asarray(
            [float(option["probability"]) for option in options], dtype=np.float64
        )
        if not np.isfinite(probabilities).all() or abs(float(probabilities.sum()) - 1.0) > 1e-5:
            raise ValueError(f"invalid probability vector in {row['id']}")
        target_index = target_indexes[0]
        target_vector = np.zeros_like(probabilities)
        target_vector[target_index] = 1.0
        prediction = options[int(probabilities.argmax())]["class_id"]
        metrics = {
            "accuracy": float(prediction == row["target"]),
            "nll": float(-np.log(max(probabilities[target_index], 1e-15))),
            "brier": float(np.square(probabilities - target_vector).sum()),
        }
        values[str(row["id"])] = {
            "scene_group_id": str(row["scene_group_id"]),
            "question_type": str(row["question_type"]),
            "taxonomy": str(row["taxonomy"]),
            **metrics,
        }
    return values


def _paired_scene_bootstrap(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    question_type: str,
    metric: str,
    left_name: str,
    right_name: str,
) -> dict[str, Any]:
    paired_ids = sorted(set(left) & set(right))
    per_scene: dict[str, list[float]] = defaultdict(list)
    for sample_id in paired_ids:
        left_row, right_row = left[sample_id], right[sample_id]
        if left_row["question_type"] != right_row["question_type"]:
            raise ValueError(f"paired task mismatch for {sample_id}")
        if left_row["question_type"] != question_type:
            continue
        if left_row["scene_group_id"] != right_row["scene_group_id"]:
            raise ValueError(f"paired scene mismatch for {sample_id}")
        per_scene[left_row["scene_group_id"]].append(
            float(left_row[metric]) - float(right_row[metric])
        )
    scenes = sorted(per_scene)
    if len(scenes) < 2:
        raise ValueError(f"too few paired scenes for {question_type}/{metric}")
    scene_differences = np.asarray(
        [np.mean(per_scene[scene]) for scene in scenes], dtype=np.float64
    )
    rng = np.random.default_rng(SEED)
    bootstrap = rng.choice(
        scene_differences,
        size=(5000, len(scene_differences)),
        replace=True,
    ).mean(axis=1)
    return {
        "left_minus_right": f"{left_name} - {right_name}",
        "metric": metric,
        "question_type": question_type,
        "scene_count": len(scenes),
        "difference": float(scene_differences.mean()),
        "percentile_95_ci": [
            float(np.quantile(bootstrap, 0.025, method="linear")),
            float(np.quantile(bootstrap, 0.975, method="linear")),
        ],
        "bootstrap_repetitions": 5000,
        "bootstrap_seed": SEED,
        "unit": "scene_group_id",
    }


def _run_variant(
    variant: str,
    vocab: VideoVocab,
    train: EncodedVideoRows,
    validation: EncodedVideoRows,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    output_dir: Path,
    device: torch.device,
    epoch_orders: list[np.ndarray],
    *,
    epochs: int,
    batch_size: int,
    validation_interval_epochs: int,
) -> dict[str, Any]:
    variant_started_utc = datetime.now(UTC).isoformat()
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        _reset_peak_memory_stats(device)
    random.seed(SEED)
    np.random.seed(SEED)
    model = TextFreeVideoReadout(vocab, temporal_mode=variant).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train = _move(train, device)
    validation = _move(validation, device)
    history_path = output_dir / f"training-history-{variant}.jsonl"
    best_score = math.inf
    best_epoch: int | None = None
    best_state: dict[str, Tensor] | None = None
    best_validation_logits: Tensor | None = None
    best_train_metrics: dict[str, Any] | None = None
    started = time.perf_counter()
    training_seconds = 0.0
    evaluation_seconds = 0.0
    with history_path.open("w", encoding="utf-8", newline="\n") as history:
        for epoch, order in enumerate(epoch_orders, start=1):
            model.train()
            epoch_started = time.perf_counter()
            grad_norms: list[float] = []
            loss_total = 0.0
            for start in range(0, len(order), batch_size):
                batch_indexes = torch.tensor(
                    order[start : start + batch_size].copy(), dtype=torch.long, device=device
                )
                batch = _subset(train, batch_indexes)
                optimizer.zero_grad(set_to_none=True)
                logits = _logits(model, batch)
                loss = F.cross_entropy(logits.float(), batch.targets)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"non-finite loss in {variant} epoch {epoch}")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                if not bool(torch.isfinite(norm)) or any(
                    parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
                    for parameter in model.parameters()
                ):
                    raise FloatingPointError(f"non-finite gradient in {variant} epoch {epoch}")
                optimizer.step()
                grad_norms.append(float(norm.item()))
                loss_total += float(loss.detach().item()) * len(batch.targets)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            training_seconds += time.perf_counter() - epoch_started
            if epoch % validation_interval_epochs:
                continue
            eval_started = time.perf_counter()
            train_result = _evaluate(model, train, train_rows)
            validation_result = _evaluate(model, validation, validation_rows)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            evaluation_seconds += time.perf_counter() - eval_started
            selector = float(validation_result["selector_score_macro_nll"])
            history_row = {
                "epoch": epoch,
                "optimizer_updates": epoch * math.ceil(len(epoch_orders[0]) / batch_size),
                "learning_rate": LEARNING_RATE,
                "train_ce": loss_total / len(train_rows),
                "train_gradient_norm_mean": float(np.mean(grad_norms)),
                "train": {
                    key: value
                    for key, value in train_result.items()
                    if key not in {"logits", "row_indexes"}
                },
                "validation": {
                    key: value
                    for key, value in validation_result.items()
                    if key not in {"logits", "row_indexes"}
                },
            }
            history.write(json.dumps(history_row, allow_nan=False) + "\n")
            history.flush()
            if selector < best_score - 1e-12:
                best_score = selector
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                }
                best_validation_logits = validation_result["logits"].detach().cpu().clone()
                best_train_metrics = {
                    key: value
                    for key, value in train_result.items()
                    if key not in {"logits", "row_indexes"}
                }
                temp = output_dir / f"best-{variant}.pt.tmp"
                torch.save({"epoch": epoch, "state_dict": best_state}, temp)
                temp.replace(output_dir / f"best-{variant}.pt")
    if (
        best_state is None
        or best_epoch is None
        or best_validation_logits is None
        or best_train_metrics is None
    ):
        raise RuntimeError(f"no best checkpoint selected for {variant}")
    final_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    final_epoch = epochs
    final_path = output_dir / f"final-{variant}.pt.tmp"
    torch.save({"epoch": final_epoch, "state_dict": final_state}, final_path)
    final_path.replace(output_dir / f"final-{variant}.pt")

    best_model = TextFreeVideoReadout(vocab, temporal_mode=variant).to(device)
    best_payload = torch.load(
        output_dir / f"best-{variant}.pt", map_location=device, weights_only=True
    )
    best_model.load_state_dict(best_payload["state_dict"], strict=True)
    selected_logits = _model_prediction(best_model, validation)
    selected_reload_max_abs_difference = float(
        (selected_logits - best_validation_logits.to(device)).abs().max().item()
    )
    if not torch.equal(selected_logits.cpu(), best_validation_logits):
        raise RuntimeError(f"best checkpoint reload changed logits for {variant}")
    best_metrics = _evaluate(best_model, validation, validation_rows)
    _write_predictions(
        output_dir / f"selected-validation-{variant}.jsonl",
        selected_logits,
        validation,
        validation_rows,
        vocab.option_to_id,
    )

    controls: dict[str, Any] = {}
    control_videos = {
        "video_zeroed": torch.zeros_like(validation.video),
        "video_scene_shuffled": _scene_shuffled_video(validation, validation_rows),
        "temporal_order_reversed": validation.video.flip(dims=(1,)),
    }
    for control_name, control_video in control_videos.items():
        control_batch = _with_video(validation, control_video)
        control_logits = _model_prediction(best_model, control_batch)
        control_metrics = _evaluate(best_model, control_batch, validation_rows)
        prediction_path = output_dir / f"selected-validation-{variant}-{control_name}.jsonl"
        _write_predictions(
            prediction_path,
            control_logits,
            control_batch,
            validation_rows,
            vocab.option_to_id,
        )
        controls[control_name] = {
            key: value
            for key, value in control_metrics.items()
            if key not in {"logits", "row_indexes"}
        }
        controls[control_name]["prediction_sha256"] = _sha256(prediction_path)
    _write_predictions(
        output_dir / f"selected-train-{variant}.jsonl",
        _model_prediction(best_model, train),
        train,
        train_rows,
        vocab.option_to_id,
    )

    final_model = TextFreeVideoReadout(vocab, temporal_mode=variant).to(device)
    final_payload = torch.load(
        output_dir / f"final-{variant}.pt", map_location=device, weights_only=True
    )
    final_model.load_state_dict(final_payload["state_dict"], strict=True)
    final_logits = _model_prediction(final_model, validation)
    final_train_metrics = _evaluate(final_model, train, train_rows)
    final_metrics = _evaluate(final_model, validation, validation_rows)
    _write_predictions(
        output_dir / f"final-validation-{variant}.jsonl",
        final_logits,
        validation,
        validation_rows,
        vocab.option_to_id,
    )
    peak_vram = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return {
        "variant": variant,
        "started_utc": variant_started_utc,
        "finished_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "epochs_completed": epochs,
        "updates_per_epoch": math.ceil(len(epoch_orders[0]) / batch_size),
        "total_optimizer_updates": epochs * math.ceil(len(epoch_orders[0]) / batch_size),
        "best_epoch": best_epoch,
        "best_step": best_epoch * math.ceil(len(epoch_orders[0]) / batch_size),
        "selected_checkpoint_example_exposures": best_epoch * len(train_rows),
        "selected_checkpoint_unique_examples": len({row["id"] for row in train_rows}),
        "selected_checkpoint_repeat_exposures": (best_epoch - 1) * len(train_rows),
        "selected_checkpoint_unique_assets": len({row["media_sha256"] for row in train_rows}),
        "selected_checkpoint_unique_scenes": len({row["scene_group_id"] for row in train_rows}),
        "final_epoch": final_epoch,
        "selected_validation_metrics": {
            key: value
            for key, value in best_metrics.items()
            if key not in {"logits", "row_indexes"}
        },
        "selected_train_metrics": best_train_metrics,
        "selected_validation_controls": controls,
        "final_validation_metrics": {
            key: value
            for key, value in final_metrics.items()
            if key not in {"logits", "row_indexes"}
        },
        "final_train_metrics": {
            key: value
            for key, value in final_train_metrics.items()
            if key not in {"logits", "row_indexes"}
        },
        "overfitting_warning": (
            "training macro NLL decreased after the selected epoch while validation macro NLL rose"
            if final_train_metrics["macro_question_type"]["nll"]
            < best_train_metrics["macro_question_type"]["nll"]
            and final_metrics["macro_question_type"]["nll"]
            > best_metrics["macro_question_type"]["nll"]
            else None
        ),
        "trainable_parameters": parameter_count,
        "frozen_encoder_parameters": VIDEO_ENCODER_PARAMETER_COUNT,
        "isolated_video_path_parameters_prequantization": (
            VIDEO_ENCODER_PARAMETER_COUNT + parameter_count
        ),
        "model_parameter_dtype": str(next(model.parameters()).dtype),
        "selected_checkpoint_sha256": _sha256(output_dir / f"best-{variant}.pt"),
        "selected_checkpoint_bytes": (output_dir / f"best-{variant}.pt").stat().st_size,
        "final_checkpoint_sha256": _sha256(output_dir / f"final-{variant}.pt"),
        "final_checkpoint_bytes": (output_dir / f"final-{variant}.pt").stat().st_size,
        "selected_prediction_sha256": _sha256(output_dir / f"selected-validation-{variant}.jsonl"),
        "final_prediction_sha256": _sha256(output_dir / f"final-validation-{variant}.jsonl"),
        "training_seconds": training_seconds,
        "evaluation_and_serialization_seconds": evaluation_seconds,
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(peak_vram),
        "best_reload_max_abs_logit_difference": selected_reload_max_abs_difference,
        "best_reload_bitwise_identical": True,
        "history_sha256": _sha256(history_path),
    }


def _smoke(vocab: VideoVocab, train: EncodedVideoRows, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(SEED)
    batch = _move(_subset(train, torch.arange(min(4, len(train.sample_ids)))), device)
    reports = {}
    for variant in ("mean_mlp", "bilstm"):
        torch.manual_seed(SEED)
        model = TextFreeVideoReadout(vocab, temporal_mode=variant).to(device)
        logits = _logits(model, batch)
        loss = F.cross_entropy(logits.float(), batch.targets)
        loss.backward()
        if not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite {variant} smoke")
        reports[variant] = {
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "loss": float(loss.detach().item()),
            "gradient_norm": float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM).item()
            ),
        }
    return reports


def main() -> None:
    args = _args()
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    preflight = _validate_inputs(config_path, config)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = _choose_device(args.device)
    sample_dir = Path(config["source"]["sample_dir"]).resolve(strict=True)
    with np.load(
        Path(config["features"]["cache"]).resolve(strict=True), allow_pickle=False
    ) as archive:
        media_hashes = archive["media_sha256"].astype(str).tolist()
        arrays = archive["features"]
        features_by_media = {key: arrays[index].copy() for index, key in enumerate(media_hashes)}
    train_rows = _read_rows(sample_dir / "train.jsonl")
    validation_rows = _read_rows(sample_dir / "validation.jsonl")
    vocab = build_video_vocab(train_rows)
    train = encode_video_rows(train_rows, features_by_media, vocab)
    validation = encode_video_rows(validation_rows, features_by_media, vocab)
    if args.smoke_only:
        result = {
            "device": str(device),
            "seed": SEED,
            "smoke": _smoke(vocab, train, device),
            "sample_id_order_sha256": preflight["train"]["ordered_sample_id_sha256"],
            "validation_loaded_for_training": False,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required for training")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite video readout output: {output_dir}")
    epochs = int(config["comparison"].get("epochs", EPOCHS))
    batch_size = int(config["comparison"].get("batch_size", BATCH_SIZE))
    max_updates = int(config["comparison"].get("max_optimizer_updates", 2048))
    validation_interval_epochs = int(
        config["comparison"].get("validation_interval_epochs", 1)
    )
    validation_interval_updates = _validation_interval_updates(
        train_count=len(train_rows),
        batch_size=batch_size,
        validation_interval_epochs=validation_interval_epochs,
        epochs=epochs,
    )
    schedule = _training_schedule(
        train_count=len(train_rows),
        batch_size=batch_size,
        epochs=epochs,
        max_optimizer_updates=max_updates,
    )
    if device.type == "cuda":
        torch.use_deterministic_algorithms(True)
        torch.cuda.synchronize(device)
    output_dir.mkdir(parents=True, exist_ok=False)
    effective_config = {
        **config,
        "resolved": {
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy_version": np.__version__,
            "repository_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                check=True,
                text=True,
            ).stdout.strip(),
            "command": [sys.executable, *sys.argv],
            "training_schedule": {
                **schedule,
                "validation_interval_epochs": validation_interval_epochs,
                "validation_interval_optimizer_updates": validation_interval_updates,
            },
            "source_hashes": {
                "training_script_sha256": _sha256(Path(__file__).resolve()),
                "readout_implementation_sha256": _sha256(
                    Path(__file__).resolve().parents[1]
                    / "src"
                    / "tiny_omni_decision"
                    / "textfree_video.py"
                ),
                "source_manifest_sha256": _sha256(
                    Path(__file__).resolve().parents[1]
                    / "manifests"
                    / "candidates"
                    / "clevrer.yaml"
                ),
            },
        },
    }
    effective_path = output_dir / "effective-config.json"
    effective_path.write_text(
        json.dumps(effective_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    preflight_path = output_dir / "input-preflight.json"
    preflight_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    epoch_orders = [
        np.random.default_rng(SEED + epoch).permutation(len(train_rows)) for epoch in range(epochs)
    ]
    run_started = datetime.now(UTC)
    run_started_counter = time.perf_counter()
    runs = {}
    for variant in ("mean_mlp", "bilstm"):
        runs[variant] = _run_variant(
            variant,
            vocab,
            train,
            validation,
            train_rows,
            validation_rows,
            output_dir,
            device,
            epoch_orders,
            epochs=epochs,
            batch_size=batch_size,
            validation_interval_epochs=validation_interval_epochs,
        )
    selected_predictions = {
        variant: _load_row_metrics(output_dir / f"selected-validation-{variant}.jsonl")
        for variant in ("mean_mlp", "bilstm")
    }
    paired_comparisons = {
        f"{question_type}/{metric}": _paired_scene_bootstrap(
            selected_predictions["bilstm"],
            selected_predictions["mean_mlp"],
            question_type,
            metric,
            "bilstm",
            "mean_mlp",
        )
        for question_type in QUESTION_TYPES
        for metric in ("accuracy", "nll", "brier")
    }
    video_dependence: dict[str, Any] = {}
    for variant in ("mean_mlp", "bilstm"):
        aligned = selected_predictions[variant]
        shuffled = _load_row_metrics(
            output_dir / f"selected-validation-{variant}-video_scene_shuffled.jsonl"
        )
        video_dependence[variant] = {
            question_type: {
                metric: _paired_scene_bootstrap(
                    aligned,
                    shuffled,
                    question_type,
                    metric,
                    "aligned_video",
                    "scene_shuffled_video",
                )
                for metric in ("accuracy", "nll", "brier")
            }
            for question_type in QUESTION_TYPES
        }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    report = {
        "status": "PASS_TEXTFREE_VIDEO_PAIRED_FIT",
        "experiment_id": config["id"],
        "source_commit": effective_config["resolved"]["repository_commit"],
        "config_sha256": _sha256(config_path),
        "effective_config_sha256": _sha256(effective_path),
        "input_preflight_sha256": _sha256(preflight_path),
        "started_utc": run_started.isoformat(),
        "finished_utc": datetime.now(UTC).isoformat(),
        "total_wall_seconds": time.perf_counter() - run_started_counter,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "feature_extraction_seconds": preflight["feature_extraction_seconds"],
        "feature_extraction_devices": preflight["feature_extraction_devices"],
        "optimizer_updates_per_variant": schedule["total_optimizer_updates"],
        "command": [sys.executable, *sys.argv],
        "train_examples_per_epoch": len(train_rows),
        "updates_per_epoch": schedule["updates_per_epoch"],
        "validation_interval_epochs": validation_interval_epochs,
        "validation_interval_optimizer_updates": validation_interval_updates,
        "epochs_completed_per_variant": epochs,
        "batch_size": batch_size,
        "train_unique_example_count": len({row["id"] for row in train_rows}),
        "train_unique_scene_count": len({row["scene_group_id"] for row in train_rows}),
        "train_unique_media_count": len({row["media_sha256"] for row in train_rows}),
        "full_run_example_exposures_per_variant": schedule["full_run_example_exposures"],
        "unique_examples_consumed_per_variant": len({row["id"] for row in train_rows}),
        "repeat_exposures_per_variant_after_first_use": (
            schedule["full_run_example_exposures"] - len({row["id"] for row in train_rows})
        ),
        "validation_examples": len(validation_rows),
        "validation_unique_scenes": len({row["scene_group_id"] for row in validation_rows}),
        "validation_used_for_checkpoint_selection": True,
        "sealed_audit_loaded": False,
        "natural_language_question_loaded_by_model": False,
        "trainable_temporal_block_parameter_tolerance": 0.01,
        "runs": runs,
        "paired_scene_bootstrap_bilstm_minus_mean_mlp": paired_comparisons,
        "paired_scene_bootstrap_aligned_minus_shuffled_video": video_dependence,
        "bootstrap_policy": {
            "cluster": "scene_group_id",
            "repetitions": 5000,
            "seed": SEED,
            "interval": "percentile 95%, NumPy linear quantile",
            "validation_selection_effect": (
                "both selected checkpoints use this development validation; intervals are "
                "descriptive and not blind-audit inference"
            ),
        },
        "cloud_cost": 0,
    }
    report_path = output_dir / "run-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "run-report.sha256").write_text(_sha256(report_path) + "\n", encoding="ascii")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
