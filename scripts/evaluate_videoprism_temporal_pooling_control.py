"""Fit only shared temporal pooling weights for the fixed CLEVRER readout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from scripts.evaluate_clevrer_binary_reasoning_controls import _load_feature_map, _read_jsonl
from scripts.evaluate_clevrer_descriptive_order_control import _paired_scene_bootstrap
from scripts.train_frozen_text_probe import CandidateScorer, _metrics
from scripts.train_frozen_video_probe import _combine, _predict, _text_features
from tiny_omni_decision.dataset import sha256_file

MODEL_REVISION = "be719a406d563b66f0ac969e7c94bab8e997c81a"
VIDEO_CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
TEXT_REVISION = "4ca70771034acceecb2e72475f72050fcdde4ddc"
TEXT_WEIGHTS_SHA256 = "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"
READOUT_SHA256 = "f68b04a4079c537772938d27c6456841c5eed19595ab49d0f42dd014211f0dbc"
TRAIN_MANIFEST_HASHES = {
    "train.jsonl": "5cd4e5155b0161e35b10aad2c98e14afba7d46407e635bd1c08d786e1e51765d",
    "validation.jsonl": "53aa6f01447a43337a601279daf620c0df91d807773522cc92230fcab173db7c",
    "fetch-report.json": "75f0511bb94df2abd7e2734c0881af0282ed3797ec7b725cc07fa7dc8fd84a44",
}
EVALUATION_VALIDATION_SHA256 = "3bd8a90f32b8977e83e6491452906a9447af7c098103df9cdb6268a92bac37f2"
QUESTION_TYPES = ("temporal_descriptive", "static_descriptive")
BOOTSTRAP_METRICS = ("accuracy", "nll", "brier", "ece_15_bins")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-sample-dir", type=Path, required=True)
    parser.add_argument("--evaluation-sample-dir", type=Path, required=True)
    parser.add_argument("--framewise-cache", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--readout-report", type=Path, required=True)
    parser.add_argument("--train-pooled-cache", type=Path, required=True)
    parser.add_argument("--evaluation-pooled-cache", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument("--reference-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_framewise_cache(
    path: Path, report_path: Path
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "complete_videoprism_framewise_feature_cache"
        or report.get("model_revision") != MODEL_REVISION
        or report.get("checkpoint_sha256") != VIDEO_CHECKPOINT_SHA256
        or report.get("frame_count") != 8
        or report.get("feature_dimension") != 768
        or report.get("feature_cache_sha256") != sha256_file(path)
    ):
        raise ValueError("framewise feature cache provenance mismatch")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"media_sha256", "features"}:
            raise ValueError("unexpected framewise cache fields")
        hashes = [str(value) for value in archive["media_sha256"].tolist()]
        features = np.asarray(archive["features"], dtype=np.float32)
    if (
        len(hashes) != len(set(hashes))
        or features.shape != (len(hashes), 8, 768)
        or not np.isfinite(features).all()
    ):
        raise ValueError("framewise cache must contain finite unique [8,768] features")
    return dict(zip(hashes, features, strict=True)), report


def _assert_pooled_parity(
    framewise: dict[str, np.ndarray],
    pooled_cache: Path,
    expected_assets: set[str],
    label: str,
) -> float:
    pooled = _load_feature_map(pooled_cache)
    if not expected_assets <= set(framewise) or not expected_assets <= set(pooled):
        raise ValueError(f"{label} framewise and pooled media identities differ")
    max_delta = max(
        float(np.max(np.abs(framewise[key].mean(axis=0) - pooled[key])))
        for key in expected_assets
    )
    if max_delta > 1e-5:
        raise ValueError(f"{label} uniform frame pooling parity exceeds 1e-5: {max_delta}")
    return max_delta


def _content_key(row: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    question = " ".join(str(row["question"]).casefold().split())
    options = tuple(sorted(" ".join(str(item).casefold().split()) for item in row["options"]))
    return question, options


def _score_rows(
    scorer: nn.Module,
    frame_features: Tensor,
    text_features: list[Tensor],
    weights: Tensor,
    device: torch.device,
) -> list[Tensor]:
    video = _weighted_pool(frame_features, weights)
    combined = _combine(video, text_features)
    return _predict(scorer, combined, device)


def _weighted_pool(frame_features: Tensor, weights: Tensor) -> Tensor:
    if frame_features.ndim != 3 or frame_features.shape[1:] != (8, 768):
        raise ValueError("frame features must have shape [examples, 8, 768]")
    if weights.shape != (8,) or not torch.isfinite(weights).all():
        raise ValueError("temporal weights must be a finite 8-vector")
    return torch.einsum("t,ntd->nd", weights, frame_features)


def _prediction_rows(
    rows: list[dict[str, Any]], logits: list[Tensor], condition: str
) -> list[dict[str, Any]]:
    output = []
    for row, values in zip(rows, logits, strict=True):
        probabilities = torch.softmax(values.float(), dim=-1).tolist()
        output.append(
            {
                "id": str(row["id"]),
                "scene_index": int(row["scene_index"]),
                "question_type": str(row["question_type"]),
                "target": str(row["target"]),
                "target_index": list(row["options"]).index(row["target"]),
                "prediction_index": int(np.argmax(probabilities)),
                "options": list(row["options"]),
                "logits": values.tolist(),
                "probabilities": probabilities,
                "condition": condition,
            }
        )
    return output


def _by_type(rows: list[dict[str, Any]], logits: list[Tensor]) -> dict[str, Any]:
    targets = [list(row["options"]).index(row["target"]) for row in rows]
    result: dict[str, Any] = {}
    for question_type in QUESTION_TYPES:
        indexes = [i for i, row in enumerate(rows) if row["question_type"] == question_type]
        if len(indexes) != 200:
            raise ValueError(f"expected 200 {question_type} rows")
        result[question_type] = _metrics(
            [logits[i] for i in indexes], [targets[i] for i in indexes]
        )
    result["macro_question_type"] = {
        metric: sum(float(result[kind][metric]) for kind in QUESTION_TYPES) / len(QUESTION_TYPES)
        for metric in BOOTSTRAP_METRICS
    }
    return result


def _train_pool_weights(
    scorer: CandidateScorer,
    frame_features: Tensor,
    text_features: list[Tensor],
    targets: list[int],
    *,
    device: torch.device,
) -> tuple[Tensor, list[float]]:
    logits = nn.Parameter(torch.zeros(8, device=device))
    optimizer = torch.optim.Adam([logits], lr=0.05)
    widths = [item.shape[0] for item in text_features]
    max_width = max(widths)
    target_tensor = torch.tensor(targets, dtype=torch.long, device=device)
    flat_text = torch.cat(text_features).to(device)
    offsets = np.cumsum([0, *widths])
    history = []
    scorer.eval()
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)
    for _ in range(200):
        weights = torch.softmax(logits, dim=0)
        video = _weighted_pool(frame_features, weights)
        expanded_video = torch.cat(
            [video[index].expand(width, -1) for index, width in enumerate(widths)], dim=0
        )
        scores = scorer(torch.cat([expanded_video, flat_text], dim=-1))
        padded = scores.new_full((len(widths), max_width), -1e4)
        for index in range(len(widths)):
            padded[index, : widths[index]] = scores[offsets[index] : offsets[index + 1]]
        loss = F.cross_entropy(padded, target_tensor)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise FloatingPointError("temporal pooling optimization produced NaN/Inf")
        history.append(float(loss.detach()))
    return torch.softmax(logits.detach(), dim=0), history


def _main() -> None:
    args = _args()
    started_utc = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite pooling-control output: {output}")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not torch.cuda.is_available():
        raise RuntimeError("this fixed VideoPrism pooling control requires the local CUDA device")
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(4)

    train_dir = args.train_sample_dir.resolve(strict=True)
    evaluation_dir = args.evaluation_sample_dir.resolve(strict=True)
    train_rows = _read_jsonl(train_dir / "train.jsonl")
    evaluation_rows = _read_jsonl(evaluation_dir / "validation.jsonl")
    if len(train_rows) != 512 or len(evaluation_rows) != 400:
        raise ValueError("frozen task requires 512 train and 400 evaluation questions")
    if len({int(row["scene_index"]) for row in train_rows}) != 256:
        raise ValueError("expected 256 unique training scenes")
    if len({int(row["scene_index"]) for row in evaluation_rows}) != 200:
        raise ValueError("expected 200 unique evaluation scenes")
    train_scenes = {int(row["scene_index"]) for row in train_rows}
    eval_scenes = {int(row["scene_index"]) for row in evaluation_rows}
    train_assets = {str(row["media_sha256"]) for row in train_rows}
    eval_assets = {str(row["media_sha256"]) for row in evaluation_rows}
    if train_scenes & eval_scenes or train_assets & eval_assets:
        raise ValueError("train/evaluation scenes or media assets overlap")
    if sha256_file(train_dir / "train.jsonl") != TRAIN_MANIFEST_HASHES["train.jsonl"]:
        raise ValueError("frozen training manifest hash mismatch")
    if sha256_file(train_dir / "validation.jsonl") != TRAIN_MANIFEST_HASHES["validation.jsonl"]:
        raise ValueError("frozen training-corpus validation manifest hash mismatch")
    if sha256_file(train_dir / "fetch-report.json") != TRAIN_MANIFEST_HASHES["fetch-report.json"]:
        raise ValueError("frozen training-corpus fetch report hash mismatch")
    if sha256_file(evaluation_dir / "validation.jsonl") != EVALUATION_VALIDATION_SHA256:
        raise ValueError("frozen heldout validation manifest hash mismatch")

    frame_cache, frame_report = _load_framewise_cache(
        args.framewise_cache.resolve(strict=True),
        args.framewise_cache.resolve(strict=True).with_name("cache-report.json"),
    )
    if set(frame_cache) != train_assets | eval_assets:
        raise ValueError("framewise cache assets do not exactly match the frozen samples")
    train_pooled_path = args.train_pooled_cache.resolve(strict=True)
    eval_pooled_path = args.evaluation_pooled_cache.resolve(strict=True)
    train_parity = _assert_pooled_parity(
        frame_cache, train_pooled_path, train_assets, "training assets"
    )
    # Validate each reference pooled cache contains exactly its matching split.
    train_pooled = _load_feature_map(train_pooled_path)
    eval_pooled = _load_feature_map(eval_pooled_path)
    eval_parity = _assert_pooled_parity(
        frame_cache, eval_pooled_path, eval_assets, "heldout evaluation assets"
    )
    del train_pooled, eval_pooled

    readout_path = args.readout.resolve(strict=True)
    if sha256_file(readout_path) != READOUT_SHA256:
        raise ValueError("fixed readout checkpoint hash mismatch")
    readout_report_path = args.readout_report.resolve(strict=True)
    readout_report = json.loads(readout_report_path.read_text(encoding="utf-8"))
    if (
        readout_report.get("status") != "complete_clevrer_frozen_videoprism_probe"
        or readout_report.get("sample_manifest_sha256")
        != {
            "train": TRAIN_MANIFEST_HASHES["train.jsonl"],
            "validation": TRAIN_MANIFEST_HASHES["validation.jsonl"],
            "fetch_report": TRAIN_MANIFEST_HASHES["fetch-report.json"],
        }
        or readout_report.get("effective_config", {}).get("frame_count") != 8
        or readout_report.get("effective_config", {}).get("seed") != 17
    ):
        raise ValueError("fixed readout training provenance mismatch")

    text_model = args.text_model.resolve(strict=True)
    if sha256_file(text_model / "model.safetensors") != TEXT_WEIGHTS_SHA256:
        raise ValueError("MiniLM weight hash mismatch")
    reference_dir = args.reference_results.resolve(strict=True)
    reference_report = json.loads((reference_dir / "report.json").read_text(encoding="utf-8"))
    reference_report_path = reference_dir / "report.json"
    if (
        reference_report.get("status")
        != "complete_clevrer_descriptive_temporal_order_control"
        or reference_report.get("model", {}).get("readout_sha256") != READOUT_SHA256
    ):
        raise ValueError("heldout fixed readout provenance mismatch")
    reference_predictions_path = reference_dir / "predictions.jsonl"
    if sha256_file(reference_predictions_path) != reference_report["predictions"]["sha256"]:
        raise ValueError("heldout reference predictions hash mismatch")
    reference_predictions = _read_jsonl(reference_predictions_path)
    reference_forward = [row for row in reference_predictions if row.get("condition") == "forward"]
    if not reference_forward:
        raise ValueError("forward baseline predictions are missing")

    unique_frame = {key: torch.from_numpy(value.copy()) for key, value in frame_cache.items()}
    train_video = torch.stack([unique_frame[str(row["media_sha256"])] for row in train_rows]).to(
        device
    )
    eval_video = torch.stack(
        [unique_frame[str(row["media_sha256"])] for row in evaluation_rows]
    ).to(device)
    text_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(text_model, local_files_only=True)
    text_encoder = AutoModel.from_pretrained(text_model, local_files_only=True).eval().to(device)
    for parameter in text_encoder.parameters():
        parameter.requires_grad_(False)
    train_text, train_targets = _text_features(train_rows, tokenizer, text_encoder, device)
    eval_text, eval_targets = _text_features(evaluation_rows, tokenizer, text_encoder, device)
    train_text = [features.to(device) for features in train_text]
    eval_text = [features.to(device) for features in eval_text]
    text_seconds = time.perf_counter() - text_started
    del tokenizer, text_encoder
    torch.cuda.empty_cache()
    if eval_targets != [list(row["options"]).index(row["target"]) for row in evaluation_rows]:
        raise ValueError("heldout text feature targets differ from source labels")

    scorer = CandidateScorer(2304, hidden_size=128).to(device).eval()
    scorer.load_state_dict(
        torch.load(readout_path, map_location="cpu", weights_only=True), strict=True
    )
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)

    uniform = torch.full((8,), 1 / 8, device=device)
    baseline_train_logits = _score_rows(scorer, train_video, train_text, uniform, device)
    baseline_eval_logits = _score_rows(scorer, eval_video, eval_text, uniform, device)
    baseline_eval_predictions = _prediction_rows(evaluation_rows, baseline_eval_logits, "uniform")
    if [row["id"] for row in reference_forward] != [row["id"] for row in baseline_eval_predictions]:
        raise ValueError("heldout forward prediction IDs differ from frozen row order")
    max_probability_delta = max(
        abs(float(a) - float(b))
        for current, reference in zip(baseline_eval_predictions, reference_forward, strict=True)
        for a, b in zip(current["probabilities"], reference["probabilities"], strict=True)
    )
    if max_probability_delta > 1e-6:
        raise ValueError(f"uniform pooled readout parity exceeds 1e-6: {max_probability_delta}")

    training_started = time.perf_counter()
    learned_weights, loss_history = _train_pool_weights(
        scorer,
        train_video,
        train_text,
        train_targets,
        device=device,
    )
    torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - training_started
    candidate_train_logits = _score_rows(scorer, train_video, train_text, learned_weights, device)
    candidate_eval_logits = _score_rows(scorer, eval_video, eval_text, learned_weights, device)
    candidate_eval_predictions = _prediction_rows(
        evaluation_rows, candidate_eval_logits, "learned_temporal_pooling"
    )

    baseline_metrics = _by_type(evaluation_rows, baseline_eval_logits)
    candidate_metrics = _by_type(evaluation_rows, candidate_eval_logits)
    train_metrics = {
        "uniform": _metrics(baseline_train_logits, train_targets),
        "learned_temporal_pooling": _metrics(candidate_train_logits, train_targets),
    }
    baseline_uniform_preds = _prediction_rows(evaluation_rows, baseline_eval_logits, "uniform")
    bootstrap = _paired_scene_bootstrap(
        baseline_uniform_preds,
        candidate_eval_predictions,
        evaluation_rows,
        replicates=10_000,
        seed=17,
    )
    bootstrap["interval"] = "95% percentile, learned temporal pooling minus uniform pooling"
    eval_keys = {_content_key(row) for row in evaluation_rows}
    train_keys = {_content_key(row) for row in train_rows}
    overlapping_eval_rows = sum(_content_key(row) in train_keys for row in evaluation_rows)

    output.mkdir(parents=True)
    predictions_path = output / "predictions.jsonl"
    predictions_path.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False) + "\n"
            for pair in zip(
                baseline_uniform_preds, candidate_eval_predictions, strict=True
            )
            for item in pair
        ),
        encoding="utf-8",
    )
    current_root = Path(__file__).resolve().parents[1]
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=current_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "schema_version": 1,
        "status": "complete_videoprism_temporal_pooling_control",
        "scope": (
            "eight shared temporal weights over frozen frame-position features; "
            "reused development evaluation"
        ),
        "started_utc": started_utc,
        "finished_utc": datetime.now(UTC).isoformat(),
        "source_commit": source_commit,
        "sealed_audit_loaded": False,
        "base_or_text_encoder_trained": False,
        "scorer_trained_or_modified": False,
        "evaluation_metrics_used_for_training_or_selection": False,
        "training": {
            "train_questions": len(train_rows),
            "train_scenes": len(train_scenes),
            "updates": 200,
            "optimizer": "Adam",
            "learning_rate": 0.05,
            "seed": 17,
            "trainable_parameters": 8,
            "train_loss_first": loss_history[0],
            "train_loss_last": loss_history[-1],
            "metrics": train_metrics,
            "loss_history": loss_history,
            "seconds": training_seconds,
        },
        "evaluation": {
            "questions": len(evaluation_rows),
            "scenes": len(eval_scenes),
            "question_type_counts": {
                kind: sum(row["question_type"] == kind for row in evaluation_rows)
                for kind in QUESTION_TYPES
            },
            "normalized_question_options_key_overlap_unique": len(train_keys & eval_keys),
            "evaluation_rows_with_train_key_overlap": overlapping_eval_rows,
            "metrics": {"uniform": baseline_metrics, "learned_temporal_pooling": candidate_metrics},
            "scene_paired_bootstrap": bootstrap,
            "metrics_used_for_tuning": False,
            "selection": (
                "fixed final weights after exactly 200 updates; "
                "no evaluation checkpoint selection"
            ),
        },
        "weights": {
            "baseline_uniform": uniform.cpu().tolist(),
            "learned_softmax_by_temporal_position": learned_weights.cpu().tolist(),
        },
        "provenance": {
            "video_revision": MODEL_REVISION,
            "video_checkpoint_sha256": VIDEO_CHECKPOINT_SHA256,
            "text_revision": TEXT_REVISION,
            "text_weights_sha256": TEXT_WEIGHTS_SHA256,
            "fixed_readout_sha256": sha256_file(readout_path),
            "readout_training_report_sha256": sha256_file(readout_report_path),
            "framewise_cache_sha256": sha256_file(args.framewise_cache.resolve(strict=True)),
            "framewise_report_sha256": sha256_file(
                args.framewise_cache.resolve(strict=True).with_name("cache-report.json")
            ),
            "train_pooled_cache_sha256": sha256_file(train_pooled_path),
            "evaluation_pooled_cache_sha256": sha256_file(eval_pooled_path),
            "train_sample_manifests": TRAIN_MANIFEST_HASHES,
            "evaluation_validation_manifest_sha256": EVALUATION_VALIDATION_SHA256,
            "uniform_feature_parity_max_abs": max(train_parity, eval_parity),
            "uniform_forward_prediction_max_probability_delta": max_probability_delta,
            "reference_report_sha256": sha256_file(reference_report_path),
            "reference_predictions_sha256": sha256_file(reference_predictions_path),
            "source_code_sha256": {
                "evaluation_script": sha256_file(Path(__file__).resolve()),
                "framewise_extractor": sha256_file(
                    current_root / "scripts" / "extract_videoprism_framewise_features.py"
                ),
                "framewise_centering_script": sha256_file(
                    current_root / "scripts" / "center_videoprism_framewise_cache.py"
                ),
                "fixed_video_readout_trainer": sha256_file(
                    current_root / "scripts" / "train_frozen_videoprism_probe.py"
                ),
                "candidate_scorer_source": sha256_file(
                    current_root / "scripts" / "train_frozen_text_probe.py"
                ),
            },
        },
        "timing_seconds": {
            "text_encoding": text_seconds,
            "pooling_training": training_seconds,
            "total": time.perf_counter() - started,
        },
        "environment": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        },
        "predictions": {
            "path": str(predictions_path),
            "sha256": sha256_file(predictions_path),
            "bytes": predictions_path.stat().st_size,
        },
        "limitations": [
            "the fixed scorer was selected using an earlier validation split",
            "the evaluation scene set was evaluated once in the preceding frame-order diagnostic",
            "the normalized question/options templates overlap between train and evaluation",
            "this tests only a shared scalar position-weighting rule, not arbitrary "
            "temporal readouts",
        ],
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "training": report["training"],
                "evaluation": report["evaluation"],
                "weights": report["weights"],
                "timing_seconds": report["timing_seconds"],
                "environment": report["environment"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    _main()
