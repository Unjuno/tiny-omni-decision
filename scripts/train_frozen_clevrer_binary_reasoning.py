"""Train a small frozen-feature CLEVRER choice-truth Decision readout."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from scripts.train_frozen_text_probe import CandidateScorer, _metrics
from scripts.train_frozen_video_probe import _combine, _predict, _text_features
from tiny_omni_decision.clevrer import (
    REASONING_TYPES,
    clevrer_binary_choice_content_key,
    expand_binary_reasoning_choices,
)
from tiny_omni_decision.dataset import sha256_file

QUESTION_TYPES = ("explanatory", "predictive", "counterfactual")
FEATURE_DIM = 768
TEXT_REVISION = "4ca70771034acceecb2e72475f72050fcdde4ddc"
VIDEO_REVISION = "be719a406d563b66f0ac969e7c94bab8e997c81a"
VIDEO_SOURCE_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
VIDEO_CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
TEXT_WEIGHTS_SHA256 = "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pretrained_reuse/path_a_videoprism_clevrer_binary_reasoning.yaml"),
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _ordered_ids_sha256(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256("\n".join(str(row["id"]) for row in rows).encode()).hexdigest()


def _task_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {kind: sum(row["question_type"] == kind for row in rows) for kind in QUESTION_TYPES}


def _validation_was_previously_used(config: dict[str, Any]) -> bool:
    return bool(config.get("source", {}).get("validation_previously_used", True))


def _validate_fetch_report(
    fetch_report: dict[str, Any],
    *,
    source_revision: str,
    train_scenes: set[int],
    validation_scenes: set[int],
    train_media: set[str],
    validation_media: set[str],
) -> None:
    if fetch_report.get("revision", fetch_report.get("source_revision")) != source_revision:
        raise ValueError("sample fetch report revision mismatch")
    scene_overlap = fetch_report.get("scene_overlap", len(train_scenes & validation_scenes))
    media_overlap = fetch_report.get("media_hash_overlap")
    if media_overlap is None:
        partition_overlaps = fetch_report.get("cross_partition_media_hash_overlap", {})
        media_overlap = sum(len(values) for values in partition_overlaps.values())
    if scene_overlap != 0 or media_overlap != 0:
        raise ValueError("sample fetch report has scene/media overlap")
    if train_scenes & validation_scenes or train_media & validation_media:
        raise ValueError("sample manifests overlap despite fetch report")


def _macro(metrics_by_type: dict[str, dict[str, Any]]) -> dict[str, float]:
    keys = ("accuracy", "nll", "brier", "ece_15_bins")
    return {
        key: sum(float(metrics_by_type[kind][key]) for kind in QUESTION_TYPES) / len(QUESTION_TYPES)
        for key in keys
    }


def _decision_metrics(
    logits: list[Tensor], targets: list[int], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    by_type: dict[str, dict[str, Any]] = {}
    for kind in QUESTION_TYPES:
        indexes = [index for index, row in enumerate(rows) if row["question_type"] == kind]
        if not indexes:
            raise ValueError(f"split contains no {kind} choices")
        by_type[kind] = {
            "count": len(indexes),
            "scene_count": len({int(rows[index]["scene_index"]) for index in indexes}),
            **_metrics([logits[index] for index in indexes], [targets[index] for index in indexes]),
        }
    return {
        "overall_example_weighted": {"count": len(rows), **_metrics(logits, targets)},
        "macro_question_type": _macro(by_type),
        "by_question_type": by_type,
    }


def _validate_inputs(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    source = config["source"]
    sample_dir = Path(source["sample_dir"]).resolve(strict=True)
    train_questions = Path(source["train_questions"]).resolve(strict=True)
    validation_questions = Path(source["validation_questions"]).resolve(strict=True)
    feature_path = Path(config["features"]["cache"]).resolve(strict=True)
    feature_report_path = Path(config["features"]["cache_report"]).resolve(strict=True)
    text_model_path = Path(config["text_encoder"]["model_path"]).resolve(strict=True)
    paths = {
        "train_manifest": sample_dir / "train.jsonl",
        "validation_manifest": sample_dir / "validation.jsonl",
        "fetch_report": sample_dir / "fetch-report.json",
        "train_questions": train_questions,
        "validation_questions": validation_questions,
        "feature_cache": feature_path,
        "feature_cache_report": feature_report_path,
        "text_model_weights": text_model_path / "model.safetensors",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"required pinned input is missing: {name}: {path}")

    if source["revision"] != "98b842082ba4f7c18b6b9e3f39145871782a65ef":
        raise ValueError("unexpected CLEVRER revision")
    if config["features"]["model_revision"] != VIDEO_REVISION:
        raise ValueError("unexpected VideoPrism revision")
    if config["features"]["source_commit"] != VIDEO_SOURCE_COMMIT:
        raise ValueError("unexpected VideoPrism source commit")
    if config["features"]["checkpoint_sha256"] != VIDEO_CHECKPOINT_SHA256:
        raise ValueError("unexpected VideoPrism checkpoint hash")
    if config["text_encoder"]["revision"] != TEXT_REVISION:
        raise ValueError("unexpected MiniLM revision")
    if config["text_encoder"]["weights_sha256"] != TEXT_WEIGHTS_SHA256:
        raise ValueError("unexpected MiniLM checkpoint hash")
    if source.get("sealed_audit_loaded") is not False:
        raise ValueError("this experiment must not load a sealed audit")

    question_hashes = {
        "train": sha256_file(train_questions),
        "validation": sha256_file(validation_questions),
    }
    if question_hashes != source["question_sha256"]:
        raise ValueError("pinned CLEVRER question JSON hash mismatch")
    if sha256_file(paths["text_model_weights"]) != TEXT_WEIGHTS_SHA256:
        raise ValueError("pinned MiniLM model.safetensors hash mismatch")

    train_manifest = _load_rows(paths["train_manifest"])
    validation_manifest = _load_rows(paths["validation_manifest"])
    if not train_manifest or not validation_manifest:
        raise ValueError("CLEVRER sample split is empty")
    if any(row.get("split") != "train" for row in train_manifest):
        raise ValueError("CLEVRER train manifest has a non-train row")
    if any(row.get("split") != "validation" for row in validation_manifest):
        raise ValueError("CLEVRER validation manifest has a non-validation row")
    train_scenes = {int(row["scene_index"]) for row in train_manifest}
    validation_scenes = {int(row["scene_index"]) for row in validation_manifest}
    train_media = {str(row["media_sha256"]) for row in train_manifest}
    validation_media = {str(row["media_sha256"]) for row in validation_manifest}
    if train_scenes & validation_scenes or train_media & validation_media:
        raise ValueError("CLEVRER scene or video media overlap across source splits")

    fetch_report = json.loads(paths["fetch_report"].read_text(encoding="utf-8"))
    _validate_fetch_report(
        fetch_report,
        source_revision=source["revision"],
        train_scenes=train_scenes,
        validation_scenes=validation_scenes,
        train_media=train_media,
        validation_media=validation_media,
    )

    feature_report = json.loads(feature_report_path.read_text(encoding="utf-8"))
    expected_report = {
        "status": "complete_videoprism_frozen_feature_cache",
        "dataset_revision": f"MIT-IBM/CLEVRER@{source['revision']}",
        "model_revision": VIDEO_REVISION,
        "upstream_source_commit": VIDEO_SOURCE_COMMIT,
        "checkpoint_sha256": VIDEO_CHECKPOINT_SHA256,
    }
    if any(feature_report.get(key) != value for key, value in expected_report.items()):
        raise ValueError("frozen VideoPrism feature-cache provenance mismatch")
    cache_hash = feature_report.get(
        "feature_file_sha256", feature_report.get("feature_cache_sha256")
    )
    if cache_hash != sha256_file(feature_path):
        raise ValueError("VideoPrism feature-cache hash mismatch")
    expected_manifests = {
        "train": sha256_file(paths["train_manifest"]),
        "validation": sha256_file(paths["validation_manifest"]),
        "fetch_report": sha256_file(paths["fetch_report"]),
    }
    if feature_report.get("sample_manifest_sha256") != expected_manifests:
        raise ValueError("VideoPrism feature cache sample manifest hashes mismatch")

    train_raw = json.loads(train_questions.read_text(encoding="utf-8"))
    validation_raw = json.loads(validation_questions.read_text(encoding="utf-8"))
    train_rows = expand_binary_reasoning_choices(train_raw, train_manifest, split="train")
    train_content = {clevrer_binary_choice_content_key(row) for row in train_rows}
    validation_rows_unfiltered = expand_binary_reasoning_choices(
        validation_raw, validation_manifest, split="validation"
    )
    overlap_before_filter = {
        clevrer_binary_choice_content_key(row) for row in validation_rows_unfiltered
    } & train_content
    validation_rows = expand_binary_reasoning_choices(
        validation_raw,
        validation_manifest,
        split="validation",
        excluded_content=train_content,
    )
    train_content = {clevrer_binary_choice_content_key(row) for row in train_rows}
    validation_content = {clevrer_binary_choice_content_key(row) for row in validation_rows}
    if train_content & validation_content:
        raise ValueError("normalized CLEVRER question/candidate overlap remains after filtering")
    if not train_rows or not validation_rows:
        raise ValueError("CLEVRER reasoning expansion produced an empty split")
    for split_name, rows in (("train", train_rows), ("validation", validation_rows)):
        if {row["question_type"] for row in rows} != REASONING_TYPES:
            raise ValueError(f"{split_name} split is missing a reasoning question type")
        for row in rows:
            if row["options"] != ["wrong", "correct"] or row["target"] not in row["options"]:
                raise ValueError(f"invalid binary Decision row: {row['id']}")

    with np.load(feature_path, allow_pickle=False) as cached:
        media_hashes = cached["media_sha256"].astype(str).tolist()
        feature_values = np.asarray(cached["features"], dtype=np.float32)
    if len(media_hashes) != len(set(media_hashes)):
        raise ValueError("VideoPrism feature cache contains duplicate media hashes")
    if (
        feature_values.shape != (len(media_hashes), FEATURE_DIM)
        or not np.isfinite(feature_values).all()
    ):
        raise ValueError("VideoPrism feature matrix has an invalid shape or non-finite values")
    feature_by_hash = dict(zip(media_hashes, feature_values, strict=True))
    expected_media = train_media | validation_media
    if set(feature_by_hash) != expected_media:
        raise ValueError("VideoPrism feature media identities differ from the frozen sample splits")

    # Verify each unique media asset once, not once per question/choice.
    checked_media: set[str] = set()
    for row in train_manifest + validation_manifest:
        media_hash = str(row["media_sha256"])
        if media_hash in checked_media:
            continue
        checked_media.add(media_hash)
        media_path = sample_dir / str(row["media_path"])
        if not media_path.is_file() or sha256_file(media_path) != media_hash:
            raise ValueError(f"missing or hash-mismatched CLEVRER media: {row['scene_index']}")

    output_paths = {name: Path(value).resolve() for name, value in config["outputs"].items()}
    counts = {
        "train": _task_counts(train_rows),
        "validation_before_content_filter": _task_counts(validation_rows_unfiltered),
        "validation": _task_counts(validation_rows),
    }
    return {
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "source_question_sha256": question_hashes,
        "sample_manifest_sha256": expected_manifests,
        "fetch_report_sha256": expected_manifests["fetch_report"],
        "feature_cache_sha256": sha256_file(feature_path),
        "feature_cache_report_sha256": sha256_file(feature_report_path),
        "text_model_weights_sha256": TEXT_WEIGHTS_SHA256,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "feature_by_hash": feature_by_hash,
        "train_scenes": train_scenes,
        "validation_scenes": validation_scenes,
        "unique_media_assets": len(expected_media),
        "content_overlap_before_filter": len(overlap_before_filter),
        "validation_rows_removed_for_exact_content_overlap": (
            len(validation_rows_unfiltered) - len(validation_rows)
        ),
        "counts": counts,
        "output_paths": output_paths,
    }


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sample_weights(rows: list[dict[str, Any]]) -> Tensor:
    scenes_by_type = {
        kind: sorted({int(row["scene_index"]) for row in rows if row["question_type"] == kind})
        for kind in QUESTION_TYPES
    }
    counts = Counter((row["question_type"], int(row["scene_index"])) for row in rows)
    values = []
    for row in rows:
        kind = str(row["question_type"])
        scene = int(row["scene_index"])
        values.append(
            1.0 / (len(QUESTION_TYPES) * len(scenes_by_type[kind]) * counts[(kind, scene)])
        )
    if not math.isclose(sum(values), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("scene/type-balanced sample weights do not sum to one")
    return torch.tensor(values, dtype=torch.float32)


def _metrics_by_scene_bootstrap(
    logits: list[Tensor],
    targets: list[int],
    rows: list[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    scene_ids = sorted({int(row["scene_index"]) for row in rows})
    type_index = {kind: index for index, kind in enumerate(QUESTION_TYPES)}
    scene_index = {scene: index for index, scene in enumerate(scene_ids)}
    counts = np.zeros((len(scene_ids), len(QUESTION_TYPES)), dtype=np.float64)
    correct_sum = np.zeros_like(counts)
    nll_sum = np.zeros_like(counts)
    brier_sum = np.zeros_like(counts)
    bin_counts = np.zeros((len(scene_ids), len(QUESTION_TYPES), 15), dtype=np.float64)
    bin_correct = np.zeros_like(bin_counts)
    bin_confidence = np.zeros_like(bin_counts)
    probabilities = torch.stack([torch.softmax(row.float(), dim=-1) for row in logits]).numpy()
    target_array = np.asarray(targets, dtype=np.int64)
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    nll = -np.log(np.clip(probabilities[np.arange(len(targets)), target_array], 1e-12, 1.0))
    onehot = np.eye(2, dtype=np.float64)[target_array]
    brier = ((probabilities - onehot) ** 2).sum(axis=1)
    correct = (predicted == target_array).astype(np.float64)
    bins = np.minimum((confidence * 15).astype(np.int64), 14)
    for index, row in enumerate(rows):
        si = scene_index[int(row["scene_index"])]
        ti = type_index[str(row["question_type"])]
        bi = bins[index]
        counts[si, ti] += 1
        correct_sum[si, ti] += correct[index]
        nll_sum[si, ti] += nll[index]
        brier_sum[si, ti] += brier[index]
        bin_counts[si, ti, bi] += 1
        bin_correct[si, ti, bi] += correct[index]
        bin_confidence[si, ti, bi] += confidence[index]

    def summarize(
        group_counts: np.ndarray,
        group_correct: np.ndarray,
        group_nll: np.ndarray,
        group_brier: np.ndarray,
        group_bins: np.ndarray,
        group_bin_correct: np.ndarray,
        group_bin_conf: np.ndarray,
    ) -> dict[str, float]:
        total = float(group_counts.sum())
        if total == 0:
            raise ValueError("empty bootstrap sample")
        ece = 0.0
        for bin_id in range(15):
            bin_count = float(group_bins[bin_id])
            if bin_count:
                ece += (bin_count / total) * abs(
                    float(group_bin_correct[bin_id] / bin_count)
                    - float(group_bin_conf[bin_id] / bin_count)
                )
        return {
            "accuracy": float(group_correct.sum() / total),
            "nll": float(group_nll.sum() / total),
            "brier": float(group_brier.sum() / total),
            "ece_15_bins": ece,
        }

    rng = np.random.default_rng(seed)
    metric_names = ("accuracy", "nll", "brier", "ece_15_bins")
    values: dict[str, dict[str, list[float]]] = {
        "overall_example_weighted": {key: [] for key in metric_names},
        "macro_question_type": {key: [] for key in metric_names},
        **{kind: {key: [] for key in metric_names} for kind in QUESTION_TYPES},
    }
    for _ in range(replicates):
        sampled = rng.integers(0, len(scene_ids), size=len(scene_ids))
        scene_frequency = np.bincount(sampled, minlength=len(scene_ids)).astype(np.float64)
        by_type = {}
        total_counts = np.zeros(len(QUESTION_TYPES), dtype=np.float64)
        total_correct = np.zeros_like(total_counts)
        total_nll = np.zeros_like(total_counts)
        total_brier = np.zeros_like(total_counts)
        total_bins = np.zeros((len(QUESTION_TYPES), 15), dtype=np.float64)
        total_bin_correct = np.zeros_like(total_bins)
        total_bin_conf = np.zeros_like(total_bins)
        for kind, ti in type_index.items():
            type_counts = scene_frequency @ counts[:, ti]
            type_correct = scene_frequency @ correct_sum[:, ti]
            type_nll = scene_frequency @ nll_sum[:, ti]
            type_brier = scene_frequency @ brier_sum[:, ti]
            type_bins = scene_frequency @ bin_counts[:, ti, :]
            type_bin_correct = scene_frequency @ bin_correct[:, ti, :]
            type_bin_conf = scene_frequency @ bin_confidence[:, ti, :]
            summary = summarize(
                type_counts,
                type_correct,
                type_nll,
                type_brier,
                type_bins,
                type_bin_correct,
                type_bin_conf,
            )
            by_type[kind] = summary
            total_counts[ti] = type_counts
            total_correct[ti] = type_correct
            total_nll[ti] = type_nll
            total_brier[ti] = type_brier
            total_bins[ti] = type_bins
            total_bin_correct[ti] = type_bin_correct
            total_bin_conf[ti] = type_bin_conf
        overall = summarize(
            total_counts,
            total_correct,
            total_nll,
            total_brier,
            total_bins.sum(axis=0),
            total_bin_correct.sum(axis=0),
            total_bin_conf.sum(axis=0),
        )
        macro = {
            key: sum(by_type[kind][key] for kind in QUESTION_TYPES) / len(QUESTION_TYPES)
            for key in values["macro_question_type"]
        }
        for key, metric in overall.items():
            values["overall_example_weighted"][key].append(metric)
        for key, metric in macro.items():
            values["macro_question_type"][key].append(metric)
        for kind, metrics in by_type.items():
            for key, metric in metrics.items():
                values[kind][key].append(metric)

    return {
        "method": (
            "scene-cluster percentile bootstrap; sample validation scenes with "
            "replacement and keep all choices per scene"
        ),
        "replicates": replicates,
        "seed": seed,
        "independent_scenes": len(scene_ids),
        "interval": "95% percentile",
        "metrics": {
            group: {
                metric: [float(value) for value in np.quantile(samples, [0.025, 0.975])]
                for metric, samples in groups.items()
            }
            for group, groups in values.items()
        },
    }


def main() -> None:
    args = _args()
    config_path = args.config.resolve(strict=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    started_at = datetime.now(UTC).isoformat()
    preflight_started = time.perf_counter()
    checked = _validate_inputs(config_path, config)
    preflight_seconds = time.perf_counter() - preflight_started
    train_rows = checked["train_rows"]
    validation_rows = checked["validation_rows"]
    preflight = {
        "schema_version": 1,
        "status": "preflight_pass_clevrer_binary_reasoning",
        "experiment_id": config["experiment_id"],
        "source_revision": config["source"]["revision"],
        "config_sha256": checked["config_sha256"],
        "source_question_sha256": checked["source_question_sha256"],
        "sample_manifest_sha256": checked["sample_manifest_sha256"],
        "feature_cache_sha256": checked["feature_cache_sha256"],
        "feature_cache_report_sha256": checked["feature_cache_report_sha256"],
        "text_model_weights_sha256": checked["text_model_weights_sha256"],
        "train_example_count": len(train_rows),
        "validation_example_count_before_content_filter": sum(
            checked["counts"]["validation_before_content_filter"].values()
        ),
        "validation_example_count": len(validation_rows),
        "question_type_counts": checked["counts"],
        "train_scene_count": len(checked["train_scenes"]),
        "validation_scene_count": len(checked["validation_scenes"]),
        "unique_media_assets": checked["unique_media_assets"],
        "scene_overlap": 0,
        "media_hash_overlap": 0,
        "normalized_content_overlap_before_filter": checked["content_overlap_before_filter"],
        "validation_rows_removed_for_exact_content_overlap": checked[
            "validation_rows_removed_for_exact_content_overlap"
        ],
        "normalized_content_overlap_after_filter": 0,
        "train_order_sha256": _ordered_ids_sha256(train_rows),
        "validation_order_sha256": _ordered_ids_sha256(validation_rows),
        "unique_train_question_ids": len({row["parent_question_id"] for row in train_rows}),
        "unique_validation_question_ids": len(
            {row["parent_question_id"] for row in validation_rows}
        ),
        "sealed_audit_loaded": False,
        "media_hashes_checked": checked["unique_media_assets"],
        "preflight_seconds": preflight_seconds,
    }
    if args.preflight_only:
        preflight_dir = checked["output_paths"]["preflight_dir"]
        if preflight_dir.exists():
            raise FileExistsError(f"refusing to overwrite preflight output: {preflight_dir}")
        preflight_dir.mkdir(parents=True, exist_ok=False)
        _write_json(preflight_dir / "preflight.json", preflight)
        print(json.dumps(preflight, indent=2))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("this configured Path A run requires the local CUDA device")
    run_dir = checked["output_paths"]["run_dir"]
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite run output: {run_dir}")
    preflight_path = checked["output_paths"]["preflight_dir"] / "preflight.json"
    if not preflight_path.is_file():
        raise FileNotFoundError("run requires a completed --preflight-only result")
    saved_preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    for key in (
        "config_sha256",
        "source_question_sha256",
        "sample_manifest_sha256",
        "feature_cache_sha256",
        "feature_cache_report_sha256",
        "text_model_weights_sha256",
        "train_order_sha256",
        "validation_order_sha256",
        "question_type_counts",
        "normalized_content_overlap_before_filter",
        "validation_rows_removed_for_exact_content_overlap",
    ):
        if saved_preflight.get(key) != preflight.get(key):
            raise ValueError(f"saved preflight no longer matches current inputs: {key}")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "effective-config.yaml").write_bytes(config_path.read_bytes())
    _write_jsonl(run_dir / "train-binary-decisions.jsonl", train_rows)
    _write_jsonl(run_dir / "validation-binary-decisions.jsonl", validation_rows)

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(4)
    model_path = Path(config["text_encoder"]["model_path"]).resolve(strict=True)
    text_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    text_encoder = AutoModel.from_pretrained(model_path, local_files_only=True).eval().to(device)
    for parameter in text_encoder.parameters():
        parameter.requires_grad_(False)
    train_text, train_targets = _text_features(train_rows, tokenizer, text_encoder, device)
    validation_text, validation_targets = _text_features(
        validation_rows, tokenizer, text_encoder, device
    )
    text_encoding_seconds = time.perf_counter() - text_started
    del text_encoder, tokenizer
    torch.cuda.empty_cache()

    feature_by_hash = checked["feature_by_hash"]
    train_video = torch.from_numpy(
        np.stack([feature_by_hash[str(row["media_sha256"])] for row in train_rows]).copy()
    )
    validation_video = torch.from_numpy(
        np.stack([feature_by_hash[str(row["media_sha256"])] for row in validation_rows]).copy()
    )
    train_features = _combine(train_video, train_text)
    validation_features = _combine(validation_video, validation_text)
    del train_text, validation_text, train_video, validation_video, feature_by_hash
    torch.cuda.empty_cache()

    scorer = CandidateScorer(
        train_features[0].shape[-1], hidden_size=int(config["decision"]["hidden_size"])
    )
    scorer = scorer.to(device)
    optimizer = torch.optim.AdamW(
        scorer.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    weights = _sample_weights(train_rows)
    history: list[dict[str, Any]] = []
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    best_validation_macro_nll = float("inf")
    batch_size = int(config["training"]["batch_examples"])
    epochs = int(config["training"]["epochs"])
    updates_per_epoch = math.ceil(len(train_rows) / batch_size)
    generator = torch.Generator().manual_seed(seed)
    training_started = time.perf_counter()
    evaluation_seconds = 0.0
    for epoch in range(1, epochs + 1):
        scorer.train()
        order = torch.randperm(len(train_rows), generator=generator).tolist()
        epoch_losses = []
        for start in range(0, len(order), batch_size):
            indexes = order[start : start + batch_size]
            batch_features = torch.cat([train_features[index] for index in indexes]).to(device)
            batch_logits = scorer(batch_features).squeeze(-1).reshape(len(indexes), 2)
            batch_targets = torch.tensor([train_targets[index] for index in indexes], device=device)
            per_example = F.cross_entropy(batch_logits, batch_targets, reduction="none")
            batch_weights = weights[indexes].to(device)
            loss = (per_example * batch_weights).sum() * (len(train_rows) / len(indexes))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        epoch_eval_started = time.perf_counter()
        validation_logits = _predict(scorer, validation_features, device)
        validation_metrics = _decision_metrics(
            validation_logits, validation_targets, validation_rows
        )
        evaluation_seconds += time.perf_counter() - epoch_eval_started
        train_logits = _predict(scorer, train_features, device)
        train_metrics = _decision_metrics(train_logits, train_targets, train_rows)
        history.append(
            {
                "epoch": epoch,
                "optimizer_updates": epoch * updates_per_epoch,
                "train_weighted_loss_mean_batch": sum(epoch_losses) / len(epoch_losses),
                "train_metrics": train_metrics,
                "validation_metrics": validation_metrics,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        val_macro_nll = float(validation_metrics["macro_question_type"]["nll"])
        if val_macro_nll < best_validation_macro_nll:
            best_validation_macro_nll = val_macro_nll
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in scorer.state_dict().items()
            }
        del train_logits, validation_logits
    training_seconds = time.perf_counter() - training_started
    if best_state is None:
        raise RuntimeError("validation did not select a CLEVRER reasoning checkpoint")
    scorer.load_state_dict(best_state, strict=True)
    scorer.eval()

    selected_train_logits = _predict(scorer, train_features, device)
    evaluation_started = time.perf_counter()
    selected_validation_logits = _predict(scorer, validation_features, device)
    torch.cuda.synchronize(device)
    selected_validation_seconds = time.perf_counter() - evaluation_started
    train_metrics = _decision_metrics(selected_train_logits, train_targets, train_rows)
    validation_metrics = _decision_metrics(
        selected_validation_logits, validation_targets, validation_rows
    )
    bootstrap = _metrics_by_scene_bootstrap(
        selected_validation_logits,
        validation_targets,
        validation_rows,
        replicates=int(config["training"]["bootstrap_replicates"]),
        seed=int(config["training"]["bootstrap_seed"]),
    )
    best_readout_path = run_dir / "best-readout.pt"
    torch.save(best_state, best_readout_path)
    reloaded_state = torch.load(best_readout_path, map_location="cpu", weights_only=True)
    reloaded_scorer = CandidateScorer(
        train_features[0].shape[-1], hidden_size=int(config["decision"]["hidden_size"])
    ).to(device)
    reloaded_scorer.load_state_dict(reloaded_state, strict=True)
    reloaded_scorer.eval()
    reload_logits = _predict(reloaded_scorer, validation_features, device)
    reload_max_logit_delta = max(
        float((before.float() - after.float()).abs().max().item())
        for before, after in zip(selected_validation_logits, reload_logits, strict=True)
    )
    if reload_max_logit_delta != 0.0:
        raise ValueError(f"best readout reload changed logits: {reload_max_logit_delta}")
    selected_validation_logits = reload_logits
    scorer = reloaded_scorer
    _write_json(run_dir / "training-history.json", history)
    predictions = []
    for row, logits, target in zip(
        validation_rows, selected_validation_logits, validation_targets, strict=True
    ):
        probabilities = torch.softmax(logits.float(), dim=-1)
        prediction = int(probabilities.argmax().item())
        predictions.append(
            {
                "id": row["id"],
                "scene_index": row["scene_index"],
                "parent_question_id": row["parent_question_id"],
                "question_type": row["question_type"],
                "question": row["question"],
                "candidate_statement": row["candidate_statement"],
                "options": row["options"],
                "target_index": target,
                "prediction_index": prediction,
                "target": row["target"],
                "prediction": row["options"][prediction],
                "logits": logits.tolist(),
                "probabilities": probabilities.tolist(),
            }
        )
    _write_jsonl(run_dir / "validation-predictions.jsonl", predictions)
    training_examples_consumed = len(train_rows) * epochs
    trainable_parameters = sum(parameter.numel() for parameter in scorer.parameters())
    report = {
        "schema_version": 1,
        "status": "complete_frozen_clevrer_binary_reasoning_readout",
        "claim_scope": config.get(
            "claim_scope",
            "CLEVRER explanatory/predictive/counterfactual binary-choice task on reused "
            "development validation scenes; not a blind or general-world benchmark",
        ),
        "experiment_id": config["experiment_id"],
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "config_path": str(config_path),
        "config_sha256": checked["config_sha256"],
        "training_script_sha256": sha256_file(Path(__file__).resolve()),
        "source_question_sha256": checked["source_question_sha256"],
        "sample_manifest_sha256": checked["sample_manifest_sha256"],
        "feature_cache_sha256": checked["feature_cache_sha256"],
        "feature_cache_report_sha256": checked["feature_cache_report_sha256"],
        "preflight_report_sha256": sha256_file(preflight_path),
        "model": {
            "video_model_id": config["features"]["model_id"],
            "video_revision": VIDEO_REVISION,
            "video_checkpoint_sha256": VIDEO_CHECKPOINT_SHA256,
            "video_frames": int(config["features"]["frame_count"]),
            "video_encoder_frozen": True,
            "text_model_id": config["text_encoder"]["model_id"],
            "text_revision": TEXT_REVISION,
            "text_weights_sha256": TEXT_WEIGHTS_SHA256,
            "text_encoder_frozen": True,
            "readout_parameters": trainable_parameters,
        },
        "split_integrity": {
            "train_scenes": len(checked["train_scenes"]),
            "validation_scenes": len(checked["validation_scenes"]),
            "unique_media_assets": checked["unique_media_assets"],
            "scene_overlap": 0,
            "media_hash_overlap": 0,
            "normalized_question_candidate_overlap_before_filter": checked[
                "content_overlap_before_filter"
            ],
            "validation_rows_removed_for_exact_content_overlap": checked[
                "validation_rows_removed_for_exact_content_overlap"
            ],
            "normalized_question_candidate_overlap_after_filter": 0,
            "sealed_audit_loaded": False,
            "validation_previously_used_for_descriptive_readout_selection": (
                _validation_was_previously_used(config)
            ),
        },
        "sample_accounting": {
            "unique_train_choice_examples": len(train_rows),
            "unique_validation_choice_examples": len(validation_rows),
            "unique_train_scenes": len(checked["train_scenes"]),
            "unique_validation_scenes": len(checked["validation_scenes"]),
            "train_examples_consumed_across_all_epochs": training_examples_consumed,
            "unique_train_examples_consumed": len(train_rows),
            "repeated_train_example_presentations": training_examples_consumed - len(train_rows),
            "optimizer_updates": epochs * updates_per_epoch,
            "train_order_sha256": _ordered_ids_sha256(train_rows),
            "validation_order_sha256": _ordered_ids_sha256(validation_rows),
            "train_question_type_counts": _task_counts(train_rows),
            "validation_question_type_counts": _task_counts(validation_rows),
        },
        "effective_config": {
            "seed": seed,
            "epochs": epochs,
            "batch_examples": batch_size,
            "optimizer": "AdamW",
            "learning_rate": float(config["training"]["learning_rate"]),
            "weight_decay": float(config["training"]["weight_decay"]),
            "loss_weighting": "equal question type then equal scene within type",
            "selection_rule": "minimum validation macro question-type NLL, tie keeps earlier epoch",
            "video_encoder_frozen": True,
            "text_encoder_frozen": True,
            "no_scheduler": True,
        },
        "run_started_utc": started_at,
        "run_finished_utc": datetime.now(UTC).isoformat(),
        "preflight_seconds": preflight_seconds,
        "text_embedding_seconds": text_encoding_seconds,
        "readout_training_seconds": training_seconds,
        "readout_evaluation_seconds": evaluation_seconds + selected_validation_seconds,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "best_epoch_by_validation_macro_nll": best_epoch,
        "train_metrics_at_selected_checkpoint": train_metrics,
        "validation_metrics_at_selected_checkpoint": validation_metrics,
        "best_checkpoint_reload_max_logit_delta": reload_max_logit_delta,
        "train_validation_gap": {
            key: float(train_metrics["macro_question_type"][key])
            - float(validation_metrics["macro_question_type"][key])
            for key in ("accuracy", "nll", "brier", "ece_15_bins")
        },
        "validation_scene_bootstrap": bootstrap,
        "history": history,
        "hardware": {
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            "cuda_runtime": torch.version.cuda,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "numpy": np.__version__,
            "pyyaml": importlib.metadata.version("PyYAML"),
        },
        "artifact_hashes": {
            name: sha256_file(run_dir / name)
            for name in (
                "effective-config.yaml",
                "train-binary-decisions.jsonl",
                "validation-binary-decisions.jsonl",
                "best-readout.pt",
                "training-history.json",
                "validation-predictions.jsonl",
            )
        },
    }
    _write_json(run_dir / "report.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "best_epoch_by_validation_macro_nll": best_epoch,
                "train_metrics": train_metrics["macro_question_type"],
                "validation_metrics": validation_metrics["macro_question_type"],
                "validation_by_question_type": validation_metrics["by_question_type"],
                "trainable_parameters": trainable_parameters,
                "training_seconds": training_seconds,
                "peak_cuda_allocated_bytes": report["peak_cuda_allocated_bytes"],
                "report_path": str(run_dir / "report.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
