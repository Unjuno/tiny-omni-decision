"""Fit matched text-free temporal readouts on the frozen CLEVRER cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
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

from scripts.train_textfree_video_probe import (
    GRAD_CLIP_NORM,
    LEARNING_RATE,
    WEIGHT_DECAY,
    _logits,
    _metric_tensors,
    _model_prediction,
    _move,
    _scene_shuffled_video,
    _sha256,
    _subset,
    _with_video,
    _write_predictions,
)
from tiny_omni_decision.textfree_video import (
    EncodedVideoRows,
    TextFreeVideoReadout,
    VideoVocab,
    build_video_vocab,
    encode_video_rows,
    validate_video_splits,
)

EXPECTED_VIDEO_REVISION = "be719a406d563b66f0ac969e7c94bab8e997c81a"
EXPECTED_VIDEO_SOURCE_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
EXPECTED_VIDEO_CHECKPOINT_SHA256 = (
    "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
)
VARIANTS = ("mean_mlp", "bilstm")
TAXONOMIES = ("count", "exist", "query_color", "query_material", "query_shape")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


def _metric(logits: Tensor, targets: Tensor, option_mask: Tensor) -> dict[str, Any]:
    return _metric_tensors(logits, targets, option_mask)


def _evaluate(
    model: TextFreeVideoReadout,
    batch: EncodedVideoRows,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    model.eval()
    with torch.inference_mode():
        logits = _logits(model, batch).float()
    by_taxonomy: dict[str, Any] = {}
    taxonomy_selector = []
    for taxonomy in TAXONOMIES:
        indexes = [index for index, row in enumerate(rows) if row["taxonomy"] == taxonomy]
        if not indexes:
            continue
        selected = torch.tensor(indexes, dtype=torch.long, device=logits.device)
        metrics = _metric(
            logits.index_select(0, selected),
            batch.targets.index_select(0, selected),
            batch.option_mask.index_select(0, selected),
        )
        by_taxonomy[taxonomy] = metrics
        taxonomy_selector.append(float(metrics["nll"]))
    if not by_taxonomy:
        raise ValueError("validation contains no supported CLEVRER taxonomy groups")
    overall = _metric(logits, batch.targets, batch.option_mask)
    macro = {
        metric: float(np.mean([record[metric] for record in by_taxonomy.values()]))
        for metric in ("accuracy", "nll", "brier", "ece_15_bins", "mean_confidence")
    }
    return {
        "overall_example_weighted": overall,
        "macro_taxonomy": macro,
        "by_taxonomy": by_taxonomy,
        "selector_score_macro_taxonomy_nll": float(np.mean(taxonomy_selector)),
        "logits": logits,
    }


def _scene_bootstrap_difference(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    left_name: str,
    right_name: str,
    metric: str,
    *,
    repetitions: int = 5000,
    seed: int = 17,
) -> dict[str, Any]:
    if set(left) != set(right) or not left:
        raise ValueError("paired prediction IDs must match and be non-empty")
    scene_rows: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for sample_id in sorted(left):
        first, second = left[sample_id], right[sample_id]
        if first["scene_group_id"] != second["scene_group_id"]:
            raise ValueError(f"scene identity differs for paired sample {sample_id}")
        scene_rows[str(first["scene_group_id"])].append(
            (float(first[metric]), float(second[metric]))
        )
    scenes = sorted(scene_rows)
    if len(scenes) < 2:
        raise ValueError("paired scene bootstrap needs at least two scenes")
    rng = np.random.default_rng(seed)
    differences = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sampled = rng.choice(scenes, size=len(scenes), replace=True)
        values = [value for scene in sampled for value in scene_rows[str(scene)]]
        differences[index] = np.mean([item[0] - item[1] for item in values])
    observed = np.mean(
        [float(left[sample_id][metric]) - float(right[sample_id][metric]) for sample_id in left]
    )
    return {
        "left": left_name,
        "right": right_name,
        "metric": metric,
        "difference": float(observed),
        "percentile_95_ci": [
            float(np.quantile(differences, 0.025, method="linear")),
            float(np.quantile(differences, 0.975, method="linear")),
        ],
        "scenes": len(scenes),
        "repetitions": repetitions,
        "seed": seed,
        "cluster": "CLEVRER scene group",
    }


def _prediction_map(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        options = row["options"]
        probabilities = {
            str(option["option_id"]): float(option["probability"]) for option in options
        }
        target_id = str(row["target"])
        option_ids = [str(option["option_id"]) for option in options]
        probability = np.asarray(
            [probabilities[option_id] for option_id in option_ids], dtype=np.float64
        )
        target = option_ids.index(target_id)
        target_vector = np.zeros_like(probability)
        target_vector[target] = 1.0
        result[str(row["id"])] = {
            "scene_group_id": str(row["scene_group_id"]),
            "accuracy": float(str(row["prediction"]) == target_id),
            "nll": float(-np.log(max(probability[target], 1e-12))),
            "brier": float(np.square(probability - target_vector).sum()),
        }
    return result


def _preflight(
    config_path: Path, config: dict[str, Any]
) -> tuple[
    dict[str, Any],
    VideoVocab,
    EncodedVideoRows,
    EncodedVideoRows,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    if (
        config.get("schema_version") != 1
        or config.get("task", {}).get("question_types") != ["temporal_descriptive"]
        or config.get("task", {}).get("natural_language_question_as_input") is not False
    ):
        raise ValueError("config must define the frozen text-free temporal task")
    sample_dir = Path(config["source"]["sample_dir"]).resolve(strict=True)
    repository_root = Path(__file__).resolve().parents[1]
    source_manifest_path = repository_root / "manifests" / "candidates" / "clevrer.yaml"
    source_manifest = yaml.safe_load(source_manifest_path.read_text(encoding="utf-8"))
    if (
        source_manifest.get("revision") != config["source"].get("revision")
        or source_manifest.get("license") != "CC0-1.0"
        or source_manifest.get("derivative_model_training_allowed") is not True
        or source_manifest.get("usage") != "training"
        or source_manifest.get("trust_status") != "trusted"
        or _sha256(source_manifest_path) != config["source"].get("source_manifest_sha256")
    ):
        raise ValueError("repository CLEVRER license/training gate changed or was not admitted")
    train_rows = _read_jsonl(sample_dir / "train.jsonl")
    validation_rows = _read_jsonl(sample_dir / "validation.jsonl")
    validate_video_splits(train_rows, validation_rows)
    if any(row["question_type"] != "temporal_descriptive" for row in train_rows + validation_rows):
        raise ValueError(
            "temporal coverage corpus must contain temporal descriptive questions only"
        )
    expected_source = f"MIT-IBM/CLEVRER@{config['source']['revision']}"
    if any(row.get("source") != expected_source for row in train_rows + validation_rows):
        raise ValueError("sample rows do not match the pinned CLEVRER revision")
    for split_name, rows in (("train", train_rows), ("validation", validation_rows)):
        expected = config["source"][f"{split_name}_rows"]
        if len(rows) != expected:
            raise ValueError(f"{split_name} row count mismatch: {len(rows)} != {expected}")
    expected_hashes = config["source"]["manifest_sha256"]
    actual_hashes = {
        name: _sha256(sample_dir / name)
        for name in ("train.jsonl", "validation.jsonl", "fetch-report.json")
    }
    if actual_hashes != expected_hashes:
        raise ValueError("frozen media manifest hash mismatch")
    corpus_preflight_path = Path(config["source"]["corpus_preflight_path"]).resolve(strict=True)
    corpus_preflight = json.loads(corpus_preflight_path.read_text(encoding="utf-8"))
    if _sha256(corpus_preflight_path) != config["source"]["corpus_preflight_sha256"]:
        raise ValueError("frozen CPU corpus-preflight hash mismatch")
    if (
        corpus_preflight.get("source", {}).get("dataset") != "MIT-IBM/CLEVRER"
        or corpus_preflight.get("source", {}).get("revision") != config["source"].get("revision")
        or corpus_preflight.get("source", {}).get("license") != "CC0-1.0"
        or corpus_preflight.get("candidate_scene_count")
        != config["source"].get("candidate_scene_count")
        or corpus_preflight.get("candidate_scene_ids_sha256")
        != config["source"]["candidate_scene_ids_sha256"]
        or corpus_preflight.get("train_scene_ids_sha256")
        != config["source"]["train_scene_ids_sha256"]
        or corpus_preflight.get("validation_scene_ids_sha256")
        != config["source"]["validation_scene_ids_sha256"]
        or corpus_preflight.get("normalized_question_option_overlap") != 0
    ):
        raise ValueError("scene split or normalized-content preflight differs from config")
    fetch_report = json.loads((sample_dir / "fetch-report.json").read_text(encoding="utf-8"))
    if (
        fetch_report.get("status") != "complete_verified_clevrer_temporal_media"
        or fetch_report.get("media_sha256_overlap") != 0
        or fetch_report.get("dataset_revision") != config["source"]["revision"]
    ):
        raise ValueError("frozen media acquisition receipt failed")
    features_path = Path(config["features"]["cache"]).resolve(strict=True)
    feature_report_path = Path(config["features"]["cache_report"]).resolve(strict=True)
    feature_report = json.loads(feature_report_path.read_text(encoding="utf-8"))
    if (
        feature_report.get("status") != "complete_videoprism_framewise_feature_cache"
        or feature_report.get("model_revision") != EXPECTED_VIDEO_REVISION
        or feature_report.get("upstream_source_commit") != EXPECTED_VIDEO_SOURCE_COMMIT
        or feature_report.get("checkpoint_sha256") != EXPECTED_VIDEO_CHECKPOINT_SHA256
        or feature_report.get("frame_count") != 8
        or feature_report.get("feature_dimension") != 768
        or config["features"].get("frame_count") != 8
        or config["features"].get("feature_dimension") != 768
        or config["features"].get("model_revision") != EXPECTED_VIDEO_REVISION
        or config["features"].get("checkpoint_sha256") != EXPECTED_VIDEO_CHECKPOINT_SHA256
        or feature_report.get("feature_cache_sha256") != _sha256(features_path)
    ):
        raise ValueError("VideoPrism framewise feature provenance/hash mismatch")
    for split, filename in (("train", "train.jsonl"), ("validation", "validation.jsonl")):
        if feature_report.get("train_manifest_sha256", {}).get(filename) != actual_hashes[filename]:
            raise ValueError(f"feature cache lacks pinned {split} manifest binding")
    with np.load(features_path, allow_pickle=False) as archive:
        media_hashes = archive["media_sha256"].astype(str).tolist()
        values = archive["features"]
    if values.shape != (len(media_hashes), 8, 768) or values.dtype != np.float32:
        raise ValueError("expected finite float32 VideoPrism features [videos,8,768]")
    if len(media_hashes) != len(set(media_hashes)) or not np.isfinite(values).all():
        raise ValueError("duplicate media or non-finite VideoPrism features")
    feature_map = {key: values[index].copy() for index, key in enumerate(media_hashes)}
    train_media = {str(row["media_sha256"]) for row in train_rows}
    validation_media = {str(row["media_sha256"]) for row in validation_rows}
    if train_media | validation_media != set(feature_map):
        raise ValueError("feature cache media identities do not exactly match frozen manifests")
    if (
        set(feature_report.get("train_media_hashes", [])) != train_media
        or set(feature_report.get("evaluation_media_hashes", [])) != validation_media
    ):
        raise ValueError("feature report asset identities differ from frozen manifests")
    vocab = build_video_vocab(train_rows)
    train = encode_video_rows(train_rows, feature_map, vocab)
    validation = encode_video_rows(validation_rows, feature_map, vocab)
    report = {
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "sample_dir": str(sample_dir),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "corpus_preflight_sha256": _sha256(corpus_preflight_path),
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "unique_train_scenes": len({row["scene_group_id"] for row in train_rows}),
        "unique_validation_scenes": len({row["scene_group_id"] for row in validation_rows}),
        "unique_train_assets": len(train_media),
        "unique_validation_assets": len(validation_media),
        "question_type_counts": {"temporal_descriptive": len(train_rows) + len(validation_rows)},
        "train_taxonomy_counts": dict(
            sorted(Counter(row["taxonomy"] for row in train_rows).items())
        ),
        "validation_taxonomy_counts": dict(
            sorted(Counter(row["taxonomy"] for row in validation_rows).items())
        ),
        "train_sample_id_order_sha256": hashlib.sha256(
            "\n".join(row["id"] for row in train_rows).encode()
        ).hexdigest(),
        "validation_sample_id_order_sha256": hashlib.sha256(
            "\n".join(row["id"] for row in validation_rows).encode()
        ).hexdigest(),
        "train_scene_overlap": 0,
        "train_validation_media_overlap": 0,
        "normalized_content_overlap": 0,
        "feature_cache_sha256": feature_report["feature_cache_sha256"],
        "feature_count": len(media_hashes),
        "video_model_revision": feature_report["model_revision"],
        "video_checkpoint_sha256": feature_report["checkpoint_sha256"],
        "sealed_audit_loaded": False,
        "natural_language_question_used_as_model_input": False,
    }
    return report, vocab, train, validation, train_rows, validation_rows


def _run_variant(
    variant: str,
    vocab: VideoVocab,
    train: EncodedVideoRows,
    validation: EncodedVideoRows,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    output: Path,
    device: torch.device,
    epoch_orders: list[np.ndarray],
    batch_size: int,
) -> dict[str, Any]:
    variant_started = time.perf_counter()
    training_seconds = 0.0
    evaluation_seconds = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17) if device.type == "cuda" else None
    random.seed(17)
    model = TextFreeVideoReadout(vocab, temporal_mode=variant).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train = _move(train, device)
    validation = _move(validation, device)
    best_state = None
    best_score = math.inf
    best_epoch = None
    best_logits = None
    history_path = output / f"training-history-{variant}.jsonl"
    with history_path.open("x", encoding="utf-8", newline="\n") as history_file:
        for epoch, order in enumerate(epoch_orders, start=1):
            model.train()
            losses = []
            grad_norms = []
            started = time.perf_counter()
            for start in range(0, len(order), batch_size):
                indexes = torch.tensor(order[start : start + batch_size].copy(), device=device)
                batch = _subset(train, indexes)
                optimizer.zero_grad(set_to_none=True)
                logits = _logits(model, batch)
                loss = F.cross_entropy(logits.float(), batch.targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss at {variant}/{epoch}")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                if not torch.isfinite(norm):
                    raise FloatingPointError(f"non-finite gradient norm at {variant}/{epoch}")
                optimizer.step()
                losses.append(float(loss.detach().item()))
                grad_norms.append(float(norm.item()))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            training_seconds += time.perf_counter() - started
            evaluation_started = time.perf_counter()
            train_metrics = _evaluate(model, train, train_rows)
            validation_metrics = _evaluate(model, validation, validation_rows)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            evaluation_seconds += time.perf_counter() - evaluation_started
            score = validation_metrics["selector_score_macro_taxonomy_nll"]
            history_row = {
                "epoch": epoch,
                "optimizer_updates": sum(
                    len(item) // batch_size + bool(len(item) % batch_size)
                    for item in epoch_orders[:epoch]
                ),
                "lr": LEARNING_RATE,
                "train_ce_rolling_batches": float(np.mean(losses)),
                "train": {key: value for key, value in train_metrics.items() if key != "logits"},
                "validation": {
                    key: value for key, value in validation_metrics.items() if key != "logits"
                },
                "gradient_norm_mean": float(np.mean(grad_norms)),
                "epoch_train_seconds": time.perf_counter() - started,
                "validation_seconds": time.perf_counter() - evaluation_started,
            }
            history_file.write(json.dumps(history_row, allow_nan=False) + "\n")
            history_file.flush()
            if score < best_score - 1e-12:
                best_score = float(score)
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                best_logits = validation_metrics["logits"].detach().cpu().clone()
                temporary = output / f"best-{variant}.pt.tmp"
                torch.save({"epoch": epoch, "state_dict": best_state}, temporary)
                temporary.replace(output / f"best-{variant}.pt")
    if best_state is None or best_epoch is None or best_logits is None:
        raise RuntimeError(f"no best checkpoint selected for {variant}")
    final_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    torch.save(
        {"epoch": len(epoch_orders), "state_dict": final_state}, output / f"final-{variant}.pt"
    )
    selected_model = TextFreeVideoReadout(vocab, temporal_mode=variant).to(device)
    selected_model.load_state_dict(
        torch.load(output / f"best-{variant}.pt", map_location=device, weights_only=True)[
            "state_dict"
        ]
    )
    selected_logits = _model_prediction(selected_model, validation)
    if not torch.equal(selected_logits.cpu(), best_logits):
        raise RuntimeError(f"selected checkpoint reload changed logits for {variant}")
    selected_metrics = _evaluate(selected_model, validation, validation_rows)
    predictions_path = output / f"selected-validation-{variant}.jsonl"
    _write_predictions(
        predictions_path, selected_logits, validation, validation_rows, vocab.option_to_id
    )
    controls = {}
    for name, values in (
        ("video_zeroed", torch.zeros_like(validation.video)),
        ("video_scene_shuffled", _scene_shuffled_video(validation, validation_rows)),
        ("temporal_order_reversed", validation.video.flip(dims=(1,))),
    ):
        control_batch = _with_video(validation, values)
        control_metrics = _evaluate(selected_model, control_batch, validation_rows)
        path = output / f"selected-validation-{variant}-{name}.jsonl"
        control_logits = _model_prediction(selected_model, control_batch)
        _write_predictions(path, control_logits, control_batch, validation_rows, vocab.option_to_id)
        controls[name] = {key: value for key, value in control_metrics.items() if key != "logits"}
    return {
        "variant": variant,
        "trainable_parameter_count": parameters,
        "best_epoch": best_epoch,
        "best_optimizer_update": sum(
            len(item) // batch_size + bool(len(item) % batch_size)
            for item in epoch_orders[:best_epoch]
        ),
        "final_epoch": len(epoch_orders),
        "validation_selected_metrics": {
            key: value for key, value in selected_metrics.items() if key != "logits"
        },
        "final_validation_metrics": {
            key: value
            for key, value in _evaluate(model, validation, validation_rows).items()
            if key != "logits"
        },
        "controls": controls,
        "history_sha256": _sha256(history_path),
        "selected_checkpoint_sha256": _sha256(output / f"best-{variant}.pt"),
        "final_checkpoint_sha256": _sha256(output / f"final-{variant}.pt"),
        "selected_predictions_sha256": _sha256(predictions_path),
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "wall_seconds": time.perf_counter() - variant_started,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }


def main() -> None:
    args = _args()
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    preflight, vocab, train, validation, train_rows, validation_rows = _preflight(
        config_path, config
    )
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(4)
    if device.type == "cuda":
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if args.smoke_only:
        torch.manual_seed(17)
        batch = _move(_subset(train, torch.arange(min(4, len(train.sample_ids)))), device)
        results = {}
        for variant in VARIANTS:
            torch.manual_seed(17)
            model = TextFreeVideoReadout(vocab, temporal_mode=variant).to(device)
            logits = _logits(model, batch)
            loss = F.cross_entropy(logits.float(), batch.targets)
            loss.backward()
            results[variant] = {
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "finite_logits": bool(torch.isfinite(logits).all()),
                "finite_loss": bool(torch.isfinite(loss)),
            }
        print(
            json.dumps(
                {"device": str(device), "smoke": results, "validation_used_for_training": False},
                indent=2,
            )
        )
        return
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite temporal coverage run: {output}")
    epochs = int(config["training"]["epochs"])
    batch_size = int(config["training"]["batch_size"])
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    expected_training = {
        "seed": 17,
        "variants": list(VARIANTS),
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRAD_CLIP_NORM,
        "scheduler": "constant",
        "early_stopping": False,
        "selection": "minimum macro taxonomy validation NLL; earliest tie",
        "optimizer": "AdamW",
    }
    for key, expected in expected_training.items():
        if config["training"].get(key) != expected:
            raise ValueError(f"training config mismatch for {key}: {config['training'].get(key)!r}")
    updates_per_epoch = (len(train_rows) + batch_size - 1) // batch_size
    expected_updates = epochs * updates_per_epoch
    if expected_updates != config["training"]["expected_optimizer_updates"]:
        raise ValueError(f"optimizer update budget mismatch: {expected_updates}")
    output.mkdir(parents=True, exist_ok=False)
    effective = {
        **config,
        "resolved": {
            "source_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "training_script_sha256": _sha256(Path(__file__).resolve()),
            "readout_source_sha256": _sha256(
                Path(__file__).resolve().parents[1]
                / "src"
                / "tiny_omni_decision"
                / "textfree_video.py"
            ),
            "config_sha256": _sha256(config_path),
            "device": str(device),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "optimizer_updates": expected_updates,
            "updates_per_epoch": updates_per_epoch,
        },
    }
    (output / "effective-config.json").write_text(
        json.dumps(effective, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "input-preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    orders = [
        np.random.default_rng(17 + epoch).permutation(len(train_rows)) for epoch in range(epochs)
    ]
    started = datetime.now(UTC)
    timer = time.perf_counter()
    runs = {
        variant: _run_variant(
            variant,
            vocab,
            train,
            validation,
            train_rows,
            validation_rows,
            output,
            device,
            orders,
            batch_size,
        )
        for variant in VARIANTS
    }
    metric_differences = {}
    left = _prediction_map(output / "selected-validation-bilstm.jsonl")
    right = _prediction_map(output / "selected-validation-mean_mlp.jsonl")
    for metric in ("accuracy", "nll", "brier"):
        metric_differences[metric] = _scene_bootstrap_difference(
            left, right, "bilstm", "mean_mlp", metric
        )
    report = {
        "status": "complete_clevrer_temporal_coverage_probe",
        "experiment_id": config["id"],
        "source_commit": effective["resolved"]["source_commit"],
        "config_sha256": effective["resolved"]["config_sha256"],
        "effective_config_sha256": _sha256(output / "effective-config.json"),
        "input_preflight_sha256": _sha256(output / "input-preflight.json"),
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(UTC).isoformat(),
        "total_wall_seconds": time.perf_counter() - timer,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "seed": 17,
        "optimizer_updates_per_variant": expected_updates,
        "train_examples_per_epoch": len(train_rows),
        "unique_examples_exposed_per_variant": epochs * len(train_rows),
        "unique_training_examples_per_variant": len(train_rows),
        "validation_used_for_checkpoint_selection": True,
        "sealed_audit_loaded": False,
        "natural_language_question_used_as_model_input": False,
        "input_preflight": preflight,
        "runs": runs,
        "paired_scene_bootstrap_bilstm_minus_mean_mlp": metric_differences,
        "bootstrap": {
            "repetitions": 5000,
            "seed": 17,
            "unit": "scene_group_id",
            "interval": "percentile 95%",
        },
        "cloud_cost_usd": 0,
    }
    report_path = output / "run-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "run-report.sha256").write_text(_sha256(report_path) + "\n", encoding="ascii")
    print(
        json.dumps(
            {
                "status": report["status"],
                "output_dir": str(output),
                "updates": expected_updates,
                "wall_seconds": report["total_wall_seconds"],
                "runs": runs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
