"""Evaluate a fixed descriptive CLEVRER readout under frame-order reversal."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from scripts.evaluate_clevrer_binary_reasoning_controls import _load_feature_map, _read_jsonl
from scripts.train_frozen_text_probe import CandidateScorer
from scripts.train_frozen_video_probe import _combine, _predict, _text_features
from tiny_omni_decision.clevrer import expand_descriptive_questions
from tiny_omni_decision.dataset import sha256_file

DATASET_REVISION = "MIT-IBM/CLEVRER@98b842082ba4f7c18b6b9e3f39145871782a65ef"
SOURCE_QUESTION_SHA256 = "fdf841678a476655b906165e6ee4b0ed1516785c5ab927c8d0a4614e4e27e10d"
VIDEO_REVISION = "be719a406d563b66f0ac969e7c94bab8e997c81a"
VIDEO_SOURCE_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
VIDEO_CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
TEXT_REVISION = "4ca70771034acceecb2e72475f72050fcdde4ddc"
TEXT_WEIGHTS_SHA256 = "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"
READOUT_SHA256 = "f68b04a4079c537772938d27c6456841c5eed19595ab49d0f42dd014211f0dbc"
QUESTION_TYPES = ("temporal_descriptive", "static_descriptive")
METRICS = ("accuracy", "nll", "brier", "ece_15_bins")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--source-questions", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--forward-cache", type=Path, required=True)
    parser.add_argument("--reverse-cache", type=Path, required=True)
    parser.add_argument("--reference-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def _validate_source_rows(
    rows: list[dict[str, Any]], raw_questions: list[dict[str, Any]]
) -> dict[str, Any]:
    if not rows or any(row.get("split") != "validation" for row in rows):
        raise ValueError("expected non-empty validation-only rows")
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("duplicate validation question ID")
    types = Counter(str(row["question_type"]) for row in rows)
    if set(types) != set(QUESTION_TYPES):
        raise ValueError("expected temporal and static descriptive questions")
    scenes: dict[int, Counter[str]] = defaultdict(Counter)
    for row in rows:
        scenes[int(row["scene_index"])][str(row["question_type"])] += 1
    if len(scenes) != 200 or any(
        scene_counts != Counter({kind: 1 for kind in QUESTION_TYPES})
        for scene_counts in scenes.values()
    ):
        raise ValueError("expected one question of each type for each of 200 scenes")

    expanded = expand_descriptive_questions(raw_questions, rows)
    expected_by_id = {str(row["id"]): row for row in expanded}
    fields = (
        "question",
        "target",
        "options",
        "question_type",
        "taxonomy",
        "program",
        "scene_index",
        "media_sha256",
    )
    for row in rows:
        expected = expected_by_id.get(str(row["id"]))
        if expected is None or any(row.get(field) != expected.get(field) for field in fields):
            raise ValueError(f"sample row differs from source CLEVRER answer mapping: {row['id']}")

    temporal_widths = Counter(
        len(row["options"]) for row in rows if row["question_type"] == "temporal_descriptive"
    )
    temporal_taxonomies = Counter(
        str(row["taxonomy"]) for row in rows if row["question_type"] == "temporal_descriptive"
    )
    temporal_targets = Counter(
        str(row["target"]) for row in rows if row["question_type"] == "temporal_descriptive"
    )
    return {
        "sample_rows": len(rows),
        "scene_count": len(scenes),
        "question_type_counts": dict(types),
        "temporal_option_width_counts": dict(temporal_widths),
        "temporal_taxonomy_counts": dict(temporal_taxonomies),
        "temporal_target_counts": dict(temporal_targets),
        "source_answer_mapping_matches": len(rows),
    }


def _cache_report(path: Path, sample_dir: Path, temporal_order: str) -> dict[str, Any]:
    report_path = path.with_name("cache-report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest_hashes = {
        "train": sha256_file(sample_dir / "train.jsonl"),
        "validation": sha256_file(sample_dir / "validation.jsonl"),
        "fetch_report": sha256_file(sample_dir / "fetch-report.json"),
    }
    if (
        report.get("status") != "complete_videoprism_frozen_feature_cache"
        or report.get("model_revision") != VIDEO_REVISION
        or report.get("upstream_source_commit") != VIDEO_SOURCE_COMMIT
        or report.get("checkpoint_sha256") != VIDEO_CHECKPOINT_SHA256
        or report.get("dataset_revision") != DATASET_REVISION
        or report.get("sample_manifest_sha256") != manifest_hashes
        or report.get("feature_cache_sha256") != sha256_file(path)
        or report.get("frame_count") != 8
        or report.get("frame_size") != 288
        or report.get("feature_dimension") != 768
        or report.get("temporal_order", "forward") != temporal_order
    ):
        raise ValueError(f"{temporal_order} VideoPrism cache provenance mismatch")
    return report


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("cannot compute metrics on an empty prediction list")
    confidence = np.empty(len(rows), dtype=np.float64)
    correct = np.empty(len(rows), dtype=bool)
    nll = np.empty(len(rows), dtype=np.float64)
    brier = np.empty(len(rows), dtype=np.float64)
    for index, row in enumerate(rows):
        probabilities = np.asarray(row["probabilities"], dtype=np.float64)
        target = int(row["target_index"])
        if (
            probabilities.ndim != 1
            or len(probabilities) < 2
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or abs(float(probabilities.sum()) - 1.0) > 1e-5
            or not 0 <= target < len(probabilities)
        ):
            raise ValueError("invalid or unnormalized probabilities/target")
        confidence[index] = probabilities.max()
        correct[index] = int(probabilities.argmax()) == target
        nll[index] = -np.log(np.clip(probabilities[target], 1e-12, 1.0))
        one_hot = np.zeros_like(probabilities)
        one_hot[target] = 1.0
        brier[index] = np.square(probabilities - one_hot).sum()
    ece = 0.0
    for index in range(15):
        lower, upper = index / 15, (index + 1) / 15
        mask = (confidence >= lower) & (
            confidence < upper if index < 14 else confidence <= upper
        )
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean() - confidence[mask].mean()))
    return {
        "count": len(rows),
        "accuracy": float(correct.mean()),
        "nll": float(nll.mean()),
        "brier": float(brier.mean()),
        "ece_15_bins": ece,
        "mean_confidence": float(confidence.mean()),
    }


def _paired_scene_bootstrap(
    forward: list[dict[str, Any]],
    reverse: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if replicates < 100:
        raise ValueError("paired scene bootstrap requires at least 100 replicates")
    scenes = sorted({int(row["scene_index"]) for row in rows})
    if len(scenes) != 200 or len(forward) != len(rows) or len(reverse) != len(rows):
        raise ValueError("bootstrap inputs do not match the frozen 200-scene sample")
    indexes_by_scene_type: dict[int, dict[str, list[int]]] = {
        scene: {kind: [] for kind in QUESTION_TYPES} for scene in scenes
    }
    for index, row in enumerate(rows):
        indexes_by_scene_type[int(row["scene_index"])][str(row["question_type"])].append(index)
    if any(
        len(indexes_by_scene_type[scene][kind]) != 1
        for scene in scenes
        for kind in QUESTION_TYPES
    ):
        raise ValueError("bootstrap requires one row per scene and question type")

    rng = np.random.default_rng(seed)
    result: dict[str, Any] = {}
    for kind in QUESTION_TYPES:
        deltas = {metric: np.empty(replicates, dtype=np.float64) for metric in METRICS}
        forward_rows = [
            forward[indexes_by_scene_type[scene][kind][0]] for scene in scenes
        ]
        reverse_rows = [
            reverse[indexes_by_scene_type[scene][kind][0]] for scene in scenes
        ]
        for replicate in range(replicates):
            selection = rng.integers(0, len(scenes), size=len(scenes))
            selected_forward = [forward_rows[index] for index in selection]
            selected_reverse = [reverse_rows[index] for index in selection]
            left, right = _metrics(selected_forward), _metrics(selected_reverse)
            for metric in METRICS:
                deltas[metric][replicate] = float(right[metric]) - float(left[metric])
        result[kind] = {
            metric: [float(value) for value in np.quantile(values, [0.025, 0.975])]
            for metric, values in deltas.items()
        }

    macro_deltas = {metric: np.empty(replicates, dtype=np.float64) for metric in METRICS}
    for replicate in range(replicates):
        selection = rng.integers(0, len(scenes), size=len(scenes))
        per_type = {}
        for kind in QUESTION_TYPES:
            indexes = [indexes_by_scene_type[scenes[index]][kind][0] for index in selection]
            per_type[kind] = (
                _metrics([forward[index] for index in indexes]),
                _metrics([reverse[index] for index in indexes]),
            )
        for metric in METRICS:
            macro_deltas[metric][replicate] = sum(
                float(per_type[kind][1][metric]) - float(per_type[kind][0][metric])
                for kind in QUESTION_TYPES
            ) / len(QUESTION_TYPES)
    result["macro_question_type"] = {
        metric: [float(value) for value in np.quantile(values, [0.025, 0.975])]
        for metric, values in macro_deltas.items()
    }
    return {
        "method": "paired scene bootstrap; 200 scenes sampled with replacement",
        "replicates": replicates,
        "seed": seed,
        "independent_scenes": len(scenes),
        "interval": "95% percentile, reverse minus forward",
        "delta_intervals": result,
    }


def _prediction_rows(
    rows: list[dict[str, Any]], logits: list[torch.Tensor], condition: str
) -> list[dict[str, Any]]:
    predictions = []
    for row, values in zip(rows, logits, strict=True):
        probabilities = torch.softmax(values.float(), dim=-1).cpu().numpy().tolist()
        predictions.append(
            {
                "id": str(row["id"]),
                "scene_index": int(row["scene_index"]),
                "question_type": str(row["question_type"]),
                "taxonomy": str(row["taxonomy"]),
                "target": str(row["target"]),
                "options": list(row["options"]),
                "target_index": list(row["options"]).index(row["target"]),
                "prediction_index": int(np.argmax(probabilities)),
                "logits": values.float().tolist(),
                "probabilities": probabilities,
                "condition": condition,
            }
        )
    return predictions


def main() -> None:
    args = _args()
    started_utc = datetime.now(UTC).isoformat()
    total_started = time.perf_counter()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite order-control output: {output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    sample_dir = args.sample_dir.resolve(strict=True)
    rows = _read_jsonl(sample_dir / "validation.jsonl")
    raw_questions_path = args.source_questions.resolve(strict=True)
    if sha256_file(raw_questions_path) != SOURCE_QUESTION_SHA256:
        raise ValueError("CLEVRER raw validation question hash mismatch")
    source_audit = _validate_source_rows(rows, json.loads(raw_questions_path.read_text()))
    validation_assets = {str(row["media_sha256"]) for row in rows}
    train_rows = _read_jsonl(sample_dir / "train.jsonl")
    if not train_rows or any(row.get("split") != "train" for row in train_rows):
        raise ValueError("sample requires its frozen train extractor witness")
    train_assets = {str(row["media_sha256"]) for row in train_rows}
    if len(train_assets) != 1 or len(validation_assets) != 200 or train_assets & validation_assets:
        raise ValueError("unexpected train witness or validation media identity overlap")

    forward_path = args.forward_cache.resolve(strict=True)
    reverse_path = args.reverse_cache.resolve(strict=True)
    forward_report = _cache_report(forward_path, sample_dir, "forward")
    reverse_report = _cache_report(reverse_path, sample_dir, "reverse")
    invariant_fields = (
        "model_id",
        "model_revision",
        "upstream_source_commit",
        "checkpoint_sha256",
        "dataset_revision",
        "sample_manifest_sha256",
        "train_ids",
        "validation_ids",
        "train_media_hashes",
        "validation_media_hashes",
        "frame_count",
        "frame_size",
        "feature_dimension",
        "preprocessing",
        "pooling",
    )
    if any(forward_report.get(key) != reverse_report.get(key) for key in invariant_fields):
        raise ValueError("forward and reverse cache metadata differ outside temporal order")
    forward_map = _load_feature_map(forward_path)
    reverse_map = _load_feature_map(reverse_path)
    expected_assets = train_assets | validation_assets
    if set(forward_map) != expected_assets or set(reverse_map) != expected_assets:
        raise ValueError("feature cache assets do not exactly match the frozen sample")
    forward_video = torch.from_numpy(
        np.stack([forward_map[str(row["media_sha256"])] for row in rows]).copy()
    )
    reverse_video = torch.from_numpy(
        np.stack([reverse_map[str(row["media_sha256"])] for row in rows]).copy()
    )
    del forward_map, reverse_map

    text_model = args.text_model.resolve(strict=True)
    text_weights = text_model / "model.safetensors"
    if sha256_file(text_weights) != TEXT_WEIGHTS_SHA256:
        raise ValueError("MiniLM checkpoint hash mismatch")
    readout_path = args.readout.resolve(strict=True)
    if sha256_file(readout_path) != READOUT_SHA256:
        raise ValueError("fixed VideoPrism readout hash mismatch")
    if args.bootstrap_replicates != 10_000 or args.seed != 17:
        raise ValueError("bootstrap settings are frozen at 10,000 replicates and seed 17")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.set_num_threads(4)
    text_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(text_model, local_files_only=True)
    text_encoder = AutoModel.from_pretrained(text_model, local_files_only=True).eval().to(device)
    for parameter in text_encoder.parameters():
        parameter.requires_grad_(False)
    text_features, targets = _text_features(rows, tokenizer, text_encoder, device)
    text_seconds = time.perf_counter() - text_started
    del tokenizer, text_encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = CandidateScorer(2304, hidden_size=128).to(device)
    state = torch.load(readout_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    scorer_started = time.perf_counter()
    forward_logits = _predict(model, _combine(forward_video, text_features), device)
    reverse_logits = _predict(model, _combine(reverse_video, text_features), device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    scorer_seconds = time.perf_counter() - scorer_started
    forward_predictions = _prediction_rows(rows, forward_logits, "forward")
    reverse_predictions = _prediction_rows(rows, reverse_logits, "reverse")

    reference_dir = args.reference_results.resolve(strict=True)
    reference_report_path = reference_dir / "report.json"
    reference_report = json.loads(reference_report_path.read_text(encoding="utf-8"))
    selected = reference_report["readouts"]["videoprism-clevrer-expanded-selected"]
    reference_predictions_path = reference_dir / selected["predictions_file"]
    if (
        selected["readout_sha256"] != READOUT_SHA256
        or selected["feature_cache_sha256"] != forward_report["feature_cache_sha256"]
        or selected["predictions_sha256"] != sha256_file(reference_predictions_path)
    ):
        raise ValueError("saved forward reference does not match the fixed readout/cache")
    saved_forward = _read_jsonl(reference_predictions_path)
    if [row["id"] for row in saved_forward] != [row["id"] for row in rows]:
        raise ValueError("saved forward predictions do not match the frozen row order")
    max_forward_probability_delta = max(
        abs(float(left) - float(right))
        for current, saved in zip(forward_predictions, saved_forward, strict=True)
        for left, right in zip(current["probabilities"], saved["probabilities"], strict=True)
    )
    if max_forward_probability_delta > 1e-6:
        raise ValueError("recomputed forward scores fail saved-baseline parity")

    metrics = {}
    for kind in QUESTION_TYPES:
        indexes = [index for index, row in enumerate(rows) if row["question_type"] == kind]
        metrics[kind] = {
            "forward": _metrics([forward_predictions[index] for index in indexes]),
            "reversed": _metrics([reverse_predictions[index] for index in indexes]),
        }
        metrics[kind]["reverse_minus_forward"] = {
            metric: float(metrics[kind]["reversed"][metric])
            - float(metrics[kind]["forward"][metric])
            for metric in METRICS
        }
    macro = {}
    for condition in ("forward", "reversed"):
        per_type = [metrics[kind][condition] for kind in QUESTION_TYPES]
        macro[condition] = {
            metric: sum(float(item[metric]) for item in per_type) / len(per_type)
            for metric in METRICS
        }
    macro["reverse_minus_forward"] = {
        metric: float(macro["reversed"][metric]) - float(macro["forward"][metric])
        for metric in METRICS
    }
    bootstrap_started = time.perf_counter()
    bootstrap = _paired_scene_bootstrap(
        forward_predictions,
        reverse_predictions,
        rows,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    bootstrap_seconds = time.perf_counter() - bootstrap_started

    output.mkdir(parents=True)
    predictions_path = output / "predictions.jsonl"
    predictions_path.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False) + "\n"
            for pair in zip(forward_predictions, reverse_predictions, strict=True)
            for item in pair
        ),
        encoding="utf-8",
    )
    project_root = Path(__file__).resolve().parents[1]
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, cwd=project_root
    ).stdout.strip()
    report = {
        "schema_version": 1,
        "status": "complete_clevrer_descriptive_temporal_order_control",
        "scope": (
            "fixed-readout forward/reversed feature diagnostic on a reused development validation; "
            "not a blind evaluation or model-selection result"
        ),
        "started_utc": started_utc,
        "finished_utc": datetime.now(UTC).isoformat(),
        "source_commit": source_commit,
        "sealed_audit_loaded": False,
        "training_or_selection_performed": False,
        "selection_or_calibration_on_this_sample": False,
        "dataset_revision": DATASET_REVISION,
        "source_question_sha256": sha256_file(raw_questions_path),
        "sample_manifest_sha256": {
            name: sha256_file(sample_dir / name)
            for name in ("train.jsonl", "validation.jsonl", "fetch-report.json")
        },
        "source_alignment": source_audit,
        "model": {
            "videoprism_revision": VIDEO_REVISION,
            "videoprism_source_commit": VIDEO_SOURCE_COMMIT,
            "videoprism_checkpoint_sha256": VIDEO_CHECKPOINT_SHA256,
            "text_revision": TEXT_REVISION,
            "text_weights_sha256": TEXT_WEIGHTS_SHA256,
            "readout_sha256": READOUT_SHA256,
        },
        "control": {
            "primary_question_type": "temporal_descriptive",
            "negative_control_question_type": "static_descriptive",
            "intervention": "reverse the same 8 uniformly sampled frames",
            "frame_count": 8,
            "frame_size": 288,
            "paired_text_features": True,
            "frozen_readout": True,
        },
        "features": {
            "forward_sha256": sha256_file(forward_path),
            "forward_report_sha256": sha256_file(forward_path.with_name("cache-report.json")),
            "reverse_sha256": sha256_file(reverse_path),
            "reverse_report_sha256": sha256_file(reverse_path.with_name("cache-report.json")),
        },
        "validation_metrics": {"by_question_type": metrics, "macro_question_type": macro},
        "paired_scene_bootstrap": bootstrap,
        "baseline_reload_parity": {
            "reference_report_sha256": sha256_file(reference_report_path),
            "reference_predictions_sha256": sha256_file(reference_predictions_path),
            "max_forward_probability_delta": max_forward_probability_delta,
            "acceptance": "<=1e-6 versus saved forward predictions",
        },
        "timing_seconds": {
            "text_encoding": text_seconds,
            "paired_scorer_forward_and_reverse": scorer_seconds,
            "paired_scene_bootstrap": bootstrap_seconds,
            "total": time.perf_counter() - total_started,
        },
        "environment": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
        },
        "predictions": {
            "path": str(predictions_path),
            "sha256": sha256_file(predictions_path),
            "bytes": predictions_path.stat().st_size,
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {
        "status": report["status"],
        "validation_metrics": report["validation_metrics"],
        "paired_scene_bootstrap": bootstrap,
        "baseline_reload_parity": report["baseline_reload_parity"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
