"""Compare frozen VideoPrism Decision readouts with unique CLEVRER scene coverage."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from collections import defaultdict
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from scripts.audit_videoprism_checkpoint import CHECKPOINT_SHA256, MODEL_REVISION
from scripts.extract_frozen_videoprism_features import SOURCE_COMMIT as VIDEOPRISM_SOURCE_COMMIT
from scripts.train_frozen_text_probe import CandidateScorer, _metrics
from scripts.train_frozen_video_probe import TEXT_REVISION, _combine, _load_rows, _text_features

CONFIG_PATH = Path("configs/pretrained_reuse/clevrer_unique_coverage_v1.json")
TEXT_CHECKPOINT_SHA256 = "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"
DEFAULT_STEPS = 2048
DEFAULT_INTERVAL = 128
DEFAULT_BATCH_SIZE = 8
DEFAULT_MIN_DELTA = 1e-5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-sample", type=Path, required=True)
    parser.add_argument("--expanded-sample", type=Path, required=True)
    parser.add_argument("--baseline-feature-cache", type=Path, required=True)
    parser.add_argument("--baseline-cache-report", type=Path, required=True)
    parser.add_argument("--expanded-feature-cache", type=Path, required=True)
    parser.add_argument("--expanded-cache-report", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_row_groups(
    baseline_train: list[dict[str, Any]],
    additional_train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
) -> dict[str, Any]:
    for name, rows in (
        ("baseline_train", baseline_train),
        ("additional_train", additional_train),
        ("validation", validation),
    ):
        if not rows:
            raise ValueError(f"{name} is empty")
        if len({str(row["id"]) for row in rows}) != len(rows):
            raise ValueError(f"duplicate question IDs in {name}")
        if any(row["question_type"] not in {"static_descriptive", "temporal_descriptive"}
               for row in rows):
            raise ValueError(f"unexpected CLEVRER question type in {name}")
        if any(len(row["options"]) < 2 or row["target"] not in row["options"] for row in rows):
            raise ValueError(f"invalid target/options in {name}")
        scene_counts: dict[int, set[str]] = defaultdict(set)
        for row in rows:
            scene_counts[int(row["scene_index"])].add(str(row["media_sha256"]))
        if any(len(hashes) != 1 for hashes in scene_counts.values()):
            raise ValueError(f"one scene maps to multiple media hashes in {name}")
        if set(scene_counts) != {
            int(row["scene_index"]) for row in rows
        }:
            raise AssertionError("scene identity accounting failed")

    baseline_scenes = {int(row["scene_index"]) for row in baseline_train}
    additional_scenes = {int(row["scene_index"]) for row in additional_train}
    validation_scenes = {int(row["scene_index"]) for row in validation}
    baseline_media = {str(row["media_sha256"]) for row in baseline_train}
    additional_media = {str(row["media_sha256"]) for row in additional_train}
    validation_media = {str(row["media_sha256"]) for row in validation}
    if baseline_scenes & additional_scenes or baseline_media & additional_media:
        raise ValueError("baseline and additional training identities overlap")
    if (baseline_scenes | additional_scenes) & validation_scenes:
        raise ValueError("training and validation scene identities overlap")
    if (baseline_media | additional_media) & validation_media:
        raise ValueError("training and validation media identities overlap")
    if len({str(row["id"]) for row in baseline_train + additional_train}) != (
        len(baseline_train) + len(additional_train)
    ):
        raise ValueError("baseline and additional training question IDs overlap")

    def counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        return {
            kind: sum(row["question_type"] == kind for row in rows)
            for kind in ("static_descriptive", "temporal_descriptive")
        }

    return {
        "baseline": {
            "questions": len(baseline_train),
            "scenes": len(baseline_scenes),
            "unique_media": len(baseline_media),
            "question_type_counts": counts(baseline_train),
        },
        "additional": {
            "questions": len(additional_train),
            "scenes": len(additional_scenes),
            "unique_media": len(additional_media),
            "question_type_counts": counts(additional_train),
        },
        "candidate": {
            "questions": len(baseline_train) + len(additional_train),
            "scenes": len(baseline_scenes | additional_scenes),
            "unique_media": len(baseline_media | additional_media),
            "question_type_counts": counts(baseline_train + additional_train),
        },
        "validation": {
            "questions": len(validation),
            "scenes": len(validation_scenes),
            "unique_media": len(validation_media),
            "question_type_counts": counts(validation),
        },
        "baseline_additional_scene_overlap": 0,
        "train_validation_scene_overlap": 0,
        "train_validation_media_overlap": 0,
        "train_question_content_overlap": len(
            {
                (str(row["question"]).casefold().strip(), tuple(sorted(row["options"])))
                for row in baseline_train + additional_train
            }
            & {
                (str(row["question"]).casefold().strip(), tuple(sorted(row["options"])))
                for row in validation
            }
        ),
    }


def _load_feature_map(
    cache_path: Path,
    report_path: Path,
    sample_dir: Path,
) -> dict[str, np.ndarray]:
    report = _read_json(report_path)
    if report.get("status") != "complete_videoprism_frozen_feature_cache":
        raise ValueError(f"incomplete VideoPrism feature cache: {report_path}")
    if report.get("model_revision") != MODEL_REVISION:
        raise ValueError("VideoPrism model revision mismatch")
    if report.get("upstream_source_commit") != VIDEOPRISM_SOURCE_COMMIT:
        raise ValueError("VideoPrism upstream source commit mismatch")
    if report.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("VideoPrism checkpoint hash mismatch")
    if report.get("feature_cache_sha256") != sha256_file(cache_path):
        raise ValueError("VideoPrism feature-cache file hash mismatch")
    manifests = report.get("sample_manifest_sha256", {})
    expected = {
        name: sha256_file(sample_dir / filename)
        for name, filename in (
            ("train", "train.jsonl"),
            ("validation", "validation.jsonl"),
            ("fetch_report", "fetch-report.json"),
        )
    }
    if manifests != expected:
        raise ValueError(f"VideoPrism feature-cache manifest hashes mismatch: {report_path}")
    with np.load(cache_path, allow_pickle=False) as cache:
        hashes = cache["media_sha256"].astype(str).tolist()
        values = cache["features"]
    if values.shape != (len(hashes), 768) or not np.isfinite(values).all():
        raise ValueError("invalid VideoPrism feature cache shape or values")
    if len(hashes) != len(set(hashes)):
        raise ValueError("duplicate media identities in VideoPrism feature cache")
    return dict(zip(hashes, values, strict=True))


def _group_metrics(
    logits: list[torch.Tensor], targets: list[int], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    by_type: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_type[str(row["question_type"])].append(index)
    groups = {
        name: _metrics([logits[i] for i in indices], [targets[i] for i in indices])
        for name, indices in sorted(by_type.items())
    }
    if set(groups) != {"static_descriptive", "temporal_descriptive"}:
        raise ValueError("both static and temporal descriptive metrics are required")
    macro = {
        metric: float(np.mean([group[metric] for group in groups.values()]))
        for metric in ("accuracy", "nll", "brier", "ece_15_bins")
    }
    return {"macro": macro, "by_question_type": groups}


def _sample_batches(
    sample_count: int,
    batch_size: int,
    steps: int,
    seed: int,
) -> Iterator[list[int]]:
    if sample_count <= 0 or batch_size <= 0 or steps <= 0:
        raise ValueError("sample_count, batch_size, and steps must be positive")
    if sample_count < batch_size:
        raise ValueError("training pool must contain at least one full batch")
    if sample_count % batch_size:
        raise ValueError("training pool size must be divisible by batch_size")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(sample_count, generator=generator).tolist()
    cursor = 0
    for _ in range(steps):
        if cursor + batch_size > sample_count:
            order = torch.randperm(sample_count, generator=generator).tolist()
            cursor = 0
        yield order[cursor : cursor + batch_size]
        cursor += batch_size


def _sampler_audit(
    rows: list[dict[str, Any]], batch_size: int, steps: int, seed: int,
    prefix_presentations: int = 8192,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    consumed_ids: set[str] = set()
    consumed_scenes: set[int] = set()
    consumed_media: set[str] = set()
    prefix_ids: set[str] = set()
    prefix_scenes: set[int] = set()
    prefix_media: set[str] = set()
    prefix_types: dict[str, int] = defaultdict(int)
    presentations = 0
    prefix_count = 0
    for batch in _sample_batches(len(rows), batch_size, steps, seed):
        for index in batch:
            row = rows[index]
            identifier = str(row["id"])
            digest.update(identifier.encode("utf-8") + b"\n")
            consumed_ids.add(identifier)
            consumed_scenes.add(int(row["scene_index"]))
            consumed_media.add(str(row["media_sha256"]))
            if prefix_count < prefix_presentations:
                prefix_count += 1
                prefix_ids.add(identifier)
                prefix_scenes.add(int(row["scene_index"]))
                prefix_media.add(str(row["media_sha256"]))
                prefix_types[str(row["question_type"])] += 1
        presentations += len(batch)
    return {
        "sampler_order_sha256": digest.hexdigest(),
        "total_presentations": presentations,
        "unique_questions_consumed": len(consumed_ids),
        "repeated_presentations": presentations - len(consumed_ids),
        "unique_scenes_consumed": len(consumed_scenes),
        "unique_media_consumed": len(consumed_media),
        "prefix_presentations": prefix_count,
        "prefix_unique_questions": len(prefix_ids),
        "prefix_unique_scenes": len(prefix_scenes),
        "prefix_unique_media": len(prefix_media),
        "prefix_question_type_counts": dict(sorted(prefix_types.items())),
    }


def _tensor_state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _prediction_rows(
    rows: list[dict[str, Any]], logits: list[torch.Tensor], targets: list[int]
) -> list[dict[str, Any]]:
    predictions = []
    for row, scores, target in zip(rows, logits, targets, strict=True):
        probabilities = torch.softmax(scores.float(), dim=-1)
        options = list(row["options"])
        predictions.append(
            {
                "id": row["id"],
                "scene_index": int(row["scene_index"]),
                "media_sha256": row["media_sha256"],
                "question_type": row["question_type"],
                "taxonomy": row["taxonomy"],
                "target": options[target],
                "prediction": options[int(probabilities.argmax())],
                "options": options,
                "probabilities": probabilities.tolist(),
            }
        )
    return predictions


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _bootstrap_difference(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]],
    resamples: int = 10_000, seed: int = 17,
) -> dict[str, Any]:
    baseline_by_id = {str(row["id"]): row for row in baseline}
    candidate_by_id = {str(row["id"]): row for row in candidate}
    if set(baseline_by_id) != set(candidate_by_id):
        raise ValueError("paired bootstrap predictions have mismatched IDs")
    scene_rows: dict[int, list[str]] = defaultdict(list)
    for identifier, row in baseline_by_id.items():
        scene_rows[int(row["scene_index"])].append(identifier)
    candidate_scenes = {int(row["scene_index"]) for row in candidate}
    scenes = sorted(scene_rows)
    if not scenes or set(scenes) != candidate_scenes:
        raise ValueError("paired bootstrap scene groups differ")
    rng = np.random.default_rng(seed)

    def metric(rows: list[dict[str, Any]], key: str) -> float:
        if key == "accuracy":
            return float(np.mean([row["prediction"] == row["target"] for row in rows]))
        if key == "nll":
            return float(
                np.mean(
                    [
                        -np.log(
                            max(
                                row["probabilities"][row["options"].index(row["target"])],
                                1e-12,
                            )
                        )
                        for row in rows
                    ]
                )
            )
        raise ValueError(f"unsupported bootstrap metric: {key}")

    subsets = ["all", "static_descriptive", "temporal_descriptive"]
    differences: dict[str, dict[str, list[float]]] = {
        subset: {"accuracy": [], "nll": []} for subset in subsets
    }
    for _ in range(resamples):
        sampled = rng.choice(scenes, size=len(scenes), replace=True).tolist()
        ids = [identifier for scene in sampled for identifier in scene_rows[int(scene)]]
        base_rows = [baseline_by_id[identifier] for identifier in ids]
        candidate_rows = [candidate_by_id[identifier] for identifier in ids]
        for subset in subsets:
            if subset == "all":
                subset_base, subset_candidate = base_rows, candidate_rows
            else:
                subset_base = [row for row in base_rows if row["question_type"] == subset]
                subset_candidate = [row for row in candidate_rows if row["question_type"] == subset]
            if not subset_base or not subset_candidate:
                raise ValueError(f"paired bootstrap draw has no observations for {subset}")
            for key in differences[subset]:
                differences[subset][key].append(
                    metric(subset_candidate, key) - metric(subset_base, key)
                )
    return {
        "group": "video_scene",
        "groups": len(scenes),
        "resamples": resamples,
        "seed": seed,
        "interval": "percentile 95%, NumPy linear quantiles",
        "candidate_minus_baseline": {
            subset: {
                key: {
                    "point": metric(
                        candidate if subset == "all" else [
                            row for row in candidate if row["question_type"] == subset
                        ], key
                    ) - metric(
                        baseline if subset == "all" else [
                            row for row in baseline if row["question_type"] == subset
                        ], key
                    ),
                    "lower_95": float(np.quantile(values, 0.025, method="linear")),
                    "upper_95": float(np.quantile(values, 0.975, method="linear")),
                }
                for key, values in subset_values.items()
            }
            for subset, subset_values in differences.items()
        },
    }


def _fit_arm(
    name: str,
    train_rows: list[dict[str, Any]],
    train_features: list[torch.Tensor],
    train_targets: list[int],
    validation_rows: list[dict[str, Any]],
    validation_features: list[torch.Tensor],
    validation_targets: list[int],
    initial_state: dict[str, torch.Tensor],
    device: torch.device,
    output: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    scorer = CandidateScorer(train_features[0].shape[-1]).to(device)
    scorer.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(
        scorer.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    steps = int(config["training"]["optimizer_updates"])
    batch_size = int(config["training"]["batch_questions"])
    interval = int(config["training"]["evaluation_interval_updates"])
    seed = int(config["seed"])
    best_score = float("inf")
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    consumed_ids: set[str] = set()
    consumed_scenes: set[int] = set()
    consumed_media: set[str] = set()
    presentations = 0
    started = time.perf_counter()
    window_losses: list[float] = []
    window_grad_norms: list[float] = []
    batches = _sample_batches(len(train_rows), batch_size, steps, seed)

    for step, indexes in enumerate(batches, start=1):
        scorer.train()
        batch = [train_features[index].to(device) for index in indexes]
        batch_targets = [train_targets[index] for index in indexes]
        for index in indexes:
            consumed_ids.add(str(train_rows[index]["id"]))
            consumed_scenes.add(int(train_rows[index]["scene_index"]))
            consumed_media.add(str(train_rows[index]["media_sha256"]))
        presentations += len(indexes)
        widths = [item.shape[0] for item in batch]
        flat_scores = scorer(torch.cat(batch))
        logits = flat_scores.new_full((len(batch), max(widths)), -1e4)
        offset = 0
        for row_index, width in enumerate(widths):
            logits[row_index, :width] = flat_scores[offset : offset + width]
            offset += width
        target_tensor = torch.tensor(batch_targets, device=device)
        loss = F.cross_entropy(logits, target_tensor)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss in {name} at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(scorer.parameters(), float("inf"))
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite gradient norm in {name} at step {step}")
        optimizer.step()
        window_losses.append(float(loss.detach().cpu()))
        window_grad_norms.append(float(gradient_norm.detach().cpu()))

        if step % interval == 0 or step == steps:
            evaluation_started = time.perf_counter()
            fixed_train_logits = _predict_for_rows(scorer, train_features, device)
            validation_logits = _predict_for_rows(scorer, validation_features, device)
            train_metrics = _group_metrics(fixed_train_logits, train_targets, train_rows)
            validation_metrics = _group_metrics(
                validation_logits, validation_targets, validation_rows
            )
            score = validation_metrics["macro"]["nll"]
            record = {
                "step": step,
                "training_presentations_total": presentations,
                "training_unique_questions_consumed": len(consumed_ids),
                "training_repeated_presentations": presentations - len(consumed_ids),
                "training_unique_scenes_consumed": len(consumed_scenes),
                "training_unique_media_consumed": len(consumed_media),
                "rolling_train_ce": float(np.mean(window_losses)),
                "rolling_gradient_norm_mean": float(np.mean(window_grad_norms)),
                "rolling_gradient_norm_max": float(np.max(window_grad_norms)),
                "learning_rate": float(config["training"]["learning_rate"]),
                "fixed_train_metrics": train_metrics,
                "validation_metrics": validation_metrics,
                "selection_score_macro_nll": score,
                "best_step_before_update": best_step,
                "evaluation_seconds": time.perf_counter() - evaluation_started,
                "elapsed_training_seconds": time.perf_counter() - started,
            }
            history.append(record)
            if score < best_score - float(config["training"]["min_delta"]):
                best_score = score
                best_step = step
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in scorer.state_dict().items()
                }
            window_losses.clear()
            window_grad_norms.clear()

    if best_state is None:
        raise RuntimeError(f"no best checkpoint selected for {name}")
    final_state = {key: value.detach().cpu().clone() for key, value in scorer.state_dict().items()}
    best_path = output / f"{name}-best.pt"
    final_path = output / f"{name}-final.pt"
    torch.save(best_state, best_path)
    torch.save(final_state, final_path)

    scorer.load_state_dict(best_state, strict=True)
    best_train_logits = _predict_for_rows(scorer, train_features, device)
    best_validation_logits = _predict_for_rows(scorer, validation_features, device)
    best_metrics = _group_metrics(best_validation_logits, validation_targets, validation_rows)
    if abs(best_metrics["macro"]["nll"] - best_score) > 1e-7:
        raise AssertionError(f"best checkpoint reload metric mismatch for {name}")
    selected_history = next(record for record in history if record["step"] == best_step)
    best_train_predictions = _prediction_rows(train_rows, best_train_logits, train_targets)
    best_validation_predictions = _prediction_rows(
        validation_rows, best_validation_logits, validation_targets
    )
    _write_jsonl(output / f"{name}-best-train-predictions.jsonl", best_train_predictions)
    _write_jsonl(
        output / f"{name}-best-validation-predictions.jsonl", best_validation_predictions
    )

    scorer.load_state_dict(final_state, strict=True)
    final_train_logits = _predict_for_rows(scorer, train_features, device)
    final_validation_logits = _predict_for_rows(scorer, validation_features, device)
    final_metrics = _group_metrics(final_validation_logits, validation_targets, validation_rows)
    _write_jsonl(
        output / f"{name}-final-validation-predictions.jsonl",
        _prediction_rows(validation_rows, final_validation_logits, validation_targets),
    )
    with (output / f"{name}-history.jsonl").open("x", encoding="utf-8") as stream:
        for record in history:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    return {
        "name": name,
        "optimizer_updates": steps,
        "training_presentations": presentations,
        "training_unique_questions_consumed": len(consumed_ids),
        "training_repeated_presentations": presentations - len(consumed_ids),
        "training_unique_scenes_consumed": len(consumed_scenes),
        "training_unique_media_consumed": len(consumed_media),
        "selected_checkpoint_consumption": {
            "presentations": selected_history["training_presentations_total"],
            "unique_questions": selected_history["training_unique_questions_consumed"],
            "repeated_presentations": selected_history["training_repeated_presentations"],
            "unique_scenes": selected_history["training_unique_scenes_consumed"],
            "unique_media": selected_history["training_unique_media_consumed"],
        },
        "selected_step": best_step,
        "selected_validation_metrics": best_metrics,
        "selected_fixed_train_metrics": _group_metrics(
            best_train_logits, train_targets, train_rows
        ),
        "final_validation_metrics": final_metrics,
        "final_fixed_train_metrics": _group_metrics(
            final_train_logits, train_targets, train_rows
        ),
        "initial_state_sha256": _tensor_state_hash(initial_state),
        "best_state_sha256": _tensor_state_hash(best_state),
        "final_state_sha256": _tensor_state_hash(final_state),
        "best_checkpoint_sha256": sha256_file(best_path),
        "final_checkpoint_sha256": sha256_file(final_path),
        "best_reload_exact_prediction_parity": True,
        "training_seconds": time.perf_counter() - started,
        "history_file": f"{name}-history.jsonl",
        "best_validation_predictions_file": f"{name}-best-validation-predictions.jsonl",
        "final_validation_predictions_file": f"{name}-final-validation-predictions.jsonl",
    }


def _predict_for_rows(
    scorer: torch.nn.Module, features: list[torch.Tensor], device: torch.device
) -> list[torch.Tensor]:
    scorer.eval()
    with torch.inference_mode():
        widths = [item.shape[0] for item in features]
        flat = scorer(torch.cat(features).to(device)).float().cpu()
    output = []
    offset = 0
    for width in widths:
        output.append(flat[offset : offset + width])
        offset += width
    if offset != len(flat):
        raise AssertionError("candidate logits were not fully partitioned by question")
    return output


def _load_arm_predictions(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    args = _args()
    started_at = datetime.now(UTC).isoformat()
    baseline_dir = args.baseline_sample.resolve(strict=True)
    expanded_dir = args.expanded_sample.resolve(strict=True)
    baseline_cache = args.baseline_feature_cache.resolve(strict=True)
    baseline_report = args.baseline_cache_report.resolve(strict=True)
    expanded_cache = args.expanded_feature_cache.resolve(strict=True)
    expanded_report = args.expanded_cache_report.resolve(strict=True)
    text_model_path = args.text_model.resolve(strict=True)
    config_path = args.config.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config.get("experiment_id") != "clevrer-videoprism-unique-scene-coverage-v1":
        raise ValueError("unexpected unique-coverage experiment config")
    if config.get("model", {}).get("video_checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("pinned VideoPrism checkpoint differs from config")
    if config.get("model", {}).get("text_revision") != TEXT_REVISION:
        raise ValueError("pinned MiniLM revision differs from config")
    baseline_fetch = _read_json(baseline_dir / "fetch-report.json")
    expanded_fetch = _read_json(expanded_dir / "fetch-report.json")
    if baseline_fetch.get("revision") != expanded_fetch.get("revision"):
        raise ValueError("baseline and expanded CLEVRER revisions differ")
    if baseline_fetch.get("revision") != config["data"]["revision"]:
        raise ValueError("CLEVRER revision differs from registered config")

    baseline_train = _load_rows(baseline_dir / "train.jsonl")
    additional_train = _load_rows(expanded_dir / "train.jsonl")
    validation = _load_rows(expanded_dir / "validation.jsonl")
    counts = _validate_row_groups(baseline_train, additional_train, validation)
    expected = config["data"]
    if counts["baseline"]["scenes"] != expected["baseline_train_scenes"]:
        raise ValueError("baseline training scene count differs from frozen config")
    if counts["additional"]["scenes"] != expected["additional_train_scenes"]:
        raise ValueError("additional training scene count differs from frozen config")
    if counts["validation"]["scenes"] != expected["fresh_validation_scenes"]:
        raise ValueError("fresh validation scene count differs from frozen config")
    if counts["candidate"]["scenes"] != expected["candidate_total_train_scenes"]:
        raise ValueError("candidate training scene count differs from frozen config")
    previously_excluded = expanded_fetch.get("previous_scenes_excluded", {})
    previously_used_validation = set(previously_excluded.get("validation", []))
    validation_scene_ids = {int(row["scene_index"]) for row in validation}
    if previously_used_validation & validation_scene_ids:
        raise ValueError("fresh validation contains a previously excluded scene")

    baseline_map = _load_feature_map(baseline_cache, baseline_report, baseline_dir)
    expanded_map = _load_feature_map(expanded_cache, expanded_report, expanded_dir)
    feature_map = dict(baseline_map)
    for media_hash, vector in expanded_map.items():
        previous = feature_map.setdefault(media_hash, vector)
        if not np.array_equal(previous, vector):
            raise ValueError(f"conflicting feature values for media SHA-256 {media_hash}")
    required_hashes = {
        str(row["media_sha256"])
        for row in baseline_train + additional_train + validation
    }
    if required_hashes - set(feature_map):
        raise ValueError("one or more required media features are absent from pinned caches")
    cache_coverage = {
        "required_unique_media": len(required_hashes),
        "baseline_cache_entries": len(baseline_map),
        "expanded_cache_entries": len(expanded_map),
        "missing_media_features": 0,
        "feature_conflicts": 0,
    }

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.set_num_threads(4)

    text_weights = text_model_path / "model.safetensors"
    if not text_weights.is_file() or sha256_file(text_weights) != TEXT_CHECKPOINT_SHA256:
        raise ValueError("local MiniLM model.safetensors does not match the pinned checkpoint")

    all_train_rows = baseline_train + additional_train
    text_feature_device = torch.device("cpu")
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_path, local_files_only=True)
    text_encoder = (
        AutoModel.from_pretrained(text_model_path, local_files_only=True)
        .eval().to(text_feature_device)
    )
    for parameter in text_encoder.parameters():
        parameter.requires_grad_(False)
    text_encoder_parameters = sum(parameter.numel() for parameter in text_encoder.parameters())
    baseline_video = torch.from_numpy(
        np.stack([feature_map[str(row["media_sha256"])] for row in baseline_train]).copy()
    )
    additional_video = torch.from_numpy(
        np.stack([feature_map[str(row["media_sha256"])] for row in additional_train]).copy()
    )
    validation_video = torch.from_numpy(
        np.stack([feature_map[str(row["media_sha256"])] for row in validation]).copy()
    )
    baseline_text, baseline_targets = _text_features(
        baseline_train, text_tokenizer, text_encoder, text_feature_device
    )
    additional_text, additional_targets = _text_features(
        additional_train, text_tokenizer, text_encoder, text_feature_device
    )
    validation_text, validation_targets = _text_features(
        validation, text_tokenizer, text_encoder, text_feature_device
    )
    del text_encoder, text_tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    baseline_features = _combine(baseline_video, baseline_text)
    additional_features = _combine(additional_video, additional_text)
    validation_features = _combine(validation_video, validation_text)
    train_targets_candidate = baseline_targets + additional_targets
    candidate_features = baseline_features + additional_features

    torch.manual_seed(int(config["seed"]))
    batch_size = int(config["training"]["batch_questions"])
    steps = int(config["training"]["optimizer_updates"])
    sampler_audits = {
        "baseline": _sampler_audit(
            baseline_train, batch_size, steps, int(config["seed"])
        ),
        "expanded": _sampler_audit(
            all_train_rows, batch_size, steps, int(config["seed"])
        ),
    }
    template = CandidateScorer(candidate_features[0].shape[-1])
    initial_state = {
        key: value.detach().cpu().clone() for key, value in template.state_dict().items()
    }
    del template
    output.mkdir(parents=True, exist_ok=False)
    (output / "effective-config.json").write_bytes(config_bytes)
    preflight = {
        "status": "preflight_pass_before_readout_training",
        "experiment_id": config["experiment_id"],
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "sample_hashes": {
            "baseline_train_jsonl": sha256_file(baseline_dir / "train.jsonl"),
            "baseline_validation_jsonl": sha256_file(baseline_dir / "validation.jsonl"),
            "baseline_fetch_report": sha256_file(baseline_dir / "fetch-report.json"),
            "expanded_train_jsonl": sha256_file(expanded_dir / "train.jsonl"),
            "validation_jsonl": sha256_file(expanded_dir / "validation.jsonl"),
            "expanded_fetch_report": sha256_file(expanded_dir / "fetch-report.json"),
        },
        "feature_cache_hashes": {
            "baseline": sha256_file(baseline_cache),
            "baseline_report": sha256_file(baseline_report),
            "expanded": sha256_file(expanded_cache),
            "expanded_report": sha256_file(expanded_report),
        },
        "text_checkpoint_sha256": sha256_file(text_weights),
        "scene_and_media_accounting": counts,
        "cache_coverage": cache_coverage,
        "validation_id_order_sha256": hashlib.sha256(
            "\n".join(str(row["id"]) for row in validation).encode()
        ).hexdigest(),
        "sampler_audits": sampler_audits,
        "readout_initial_state_sha256": _tensor_state_hash(initial_state),
        "text_feature_preprocessing_device": str(text_feature_device),
        "training_device": str(device),
        "sealed_audit_loaded": False,
        "official_test_loaded": False,
    }
    (output / "preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    arms = {}
    for name, train_rows, train_features, targets in (
        ("baseline", baseline_train, baseline_features, baseline_targets),
        ("expanded", all_train_rows, candidate_features, train_targets_candidate),
    ):
        arms[name] = _fit_arm(
            name,
            train_rows,
            train_features,
            targets,
            validation,
            validation_features,
            validation_targets,
            initial_state,
            device,
            output,
            config,
        )

    if arms["baseline"]["initial_state_sha256"] != arms["expanded"]["initial_state_sha256"]:
        raise AssertionError("training arms did not share an identical readout initialization")

    baseline_predictions = _load_arm_predictions(
        output / "baseline-best-validation-predictions.jsonl"
    )
    candidate_predictions = _load_arm_predictions(
        output / "expanded-best-validation-predictions.jsonl"
    )
    paired = _bootstrap_difference(baseline_predictions, candidate_predictions)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    report = {
        "schema_version": 1,
        "status": "complete_clevrer_unique_coverage_comparison",
        "experiment_id": config["experiment_id"],
        "claim_scope": config["comparison"]["claim_scope"],
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "dataset_revision": baseline_fetch.get("revision"),
        "baseline_sample": str(baseline_dir),
        "expanded_sample": str(expanded_dir),
        "sample_hashes": {
            "baseline_train_jsonl": sha256_file(baseline_dir / "train.jsonl"),
            "baseline_fetch_report": sha256_file(baseline_dir / "fetch-report.json"),
            "expanded_train_jsonl": sha256_file(expanded_dir / "train.jsonl"),
            "validation_jsonl": sha256_file(expanded_dir / "validation.jsonl"),
            "expanded_fetch_report": sha256_file(expanded_dir / "fetch-report.json"),
        },
        "scene_and_media_accounting": counts,
        "feature_cache_coverage": cache_coverage,
        "feature_cache_hashes": {
            "baseline": sha256_file(baseline_cache),
            "baseline_report": sha256_file(baseline_report),
            "expanded": sha256_file(expanded_cache),
            "expanded_report": sha256_file(expanded_report),
        },
        "validation_ids_sha256": hashlib.sha256(
            "\n".join(str(row["id"]) for row in validation).encode()
        ).hexdigest(),
        "validation_scene_ids": sorted({int(row["scene_index"]) for row in validation}),
        "models": {
            "videoprism_revision": MODEL_REVISION,
            "videoprism_source_commit": VIDEOPRISM_SOURCE_COMMIT,
            "videoprism_checkpoint_sha256": CHECKPOINT_SHA256,
            "text_revision": TEXT_REVISION,
            "text_checkpoint_sha256": TEXT_CHECKPOINT_SHA256,
            "videoprism_parameters": 114_365_184,
            "text_parameters": text_encoder_parameters,
            "readout_parameters": sum(parameter.numel() for parameter in CandidateScorer(
                candidate_features[0].shape[-1]
            ).parameters()),
            "component_parameter_inventory_not_simultaneously_resident": {
                "baseline": 114_365_184 + text_encoder_parameters + sum(
                    parameter.numel() for parameter in CandidateScorer(
                        baseline_features[0].shape[-1]
                    ).parameters()
                ),
                "expanded": 114_365_184 + text_encoder_parameters + sum(
                    parameter.numel() for parameter in CandidateScorer(
                        candidate_features[0].shape[-1]
                    ).parameters()
                ),
            },
            "all_encoders_frozen": True,
            "new_encoder_or_fusion_parameters": 0,
        },
        "training": {
            "seed": config["seed"],
            "optimizer": "AdamW",
            "learning_rate": config["training"]["learning_rate"],
            "weight_decay": config["training"]["weight_decay"],
            "batch_questions": config["training"]["batch_questions"],
            "optimizer_updates_each_arm": config["training"]["optimizer_updates"],
            "evaluation_interval_updates": config["training"]["evaluation_interval_updates"],
            "early_stopping": False,
            "selector": config["training"]["selection"],
            "initial_state_hashes_identical": (
                arms["baseline"]["initial_state_sha256"] == arms["expanded"]["initial_state_sha256"]
            ),
            "arms": arms,
            "paired_scene_bootstrap_best_checkpoint": paired,
        },
        "runtime": {
            "device": str(device),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "numpy": np.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda" else None,
            "sealed_audit_loaded": False,
            "official_test_loaded": False,
            "cloud_cost_usd": 0,
        },
    }
    report_path = output / "result.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    report["artifact_hashes"] = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact-hashes.json"
    }
    (output / "artifact-hashes.json").write_text(
        json.dumps(report["artifact_hashes"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "baseline": arms["baseline"]["selected_validation_metrics"],
        "expanded": arms["expanded"]["selected_validation_metrics"],
        "paired": paired,
        "output": str(output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
