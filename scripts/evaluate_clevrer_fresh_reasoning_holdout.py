"""One-time paired evaluation on a frozen CLEVRER reasoning holdout."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import AutoModel, AutoTokenizer

from scripts.train_frozen_clevrer_binary_reasoning import (
    QUESTION_TYPES,
    _decision_metrics,
    _load_rows,
    _write_json,
    _write_jsonl,
)
from scripts.train_frozen_text_probe import CandidateScorer
from scripts.train_frozen_video_probe import _combine, _predict, _text_features
from tiny_omni_decision.clevrer import expand_binary_reasoning_choices
from tiny_omni_decision.dataset import sha256_file


def _metrics(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    chosen = probabilities[np.arange(len(targets)), targets]
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    ece = 0.0
    for index in range(15):
        low, high = index / 15, (index + 1) / 15
        selected = (confidence >= low) & (
            confidence < high if index < 14 else confidence <= high
        )
        if selected.any():
            ece += float(selected.mean()) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[targets]
    return {
        "accuracy": float(correct.mean()),
        "nll": float(-np.log(np.maximum(chosen, 1e-12)).mean()),
        "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ece_15_bins": ece,
    }


def _evaluate_readout(
    checkpoint: Path,
    rows: list[dict[str, Any]],
    video_by_hash: dict[str, np.ndarray],
    tokenizer: Any,
    encoder: Any,
    device: torch.device,
    hidden_size: int,
) -> tuple[list[int], np.ndarray, dict[str, Any]]:
    text_features, targets = _text_features(rows, tokenizer, encoder, device)
    videos = torch.from_numpy(
        np.stack([video_by_hash[str(row["media_sha256"])] for row in rows]).copy()
    )
    features = _combine(videos, text_features)
    del text_features, videos
    scorer = CandidateScorer(features[0].shape[-1], hidden_size=hidden_size).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    scorer.load_state_dict(state, strict=True)
    logits = _predict(scorer, features, device)
    probabilities = np.stack([torch.softmax(value.float(), dim=-1).numpy() for value in logits])
    metrics = _decision_metrics(logits, targets, rows)
    return targets, probabilities, metrics


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selected-run", type=Path, required=True)
    parser.add_argument("--legacy-run", type=Path, required=True)
    parser.add_argument("--holdout-dir", type=Path, required=True)
    parser.add_argument("--holdout-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = _args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"one-time holdout output already exists: {output}")
    config_path = args.config.resolve(strict=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    selected_run = args.selected_run.resolve(strict=True)
    legacy_run = args.legacy_run.resolve(strict=True)
    holdout_dir = args.holdout_dir.resolve(strict=True)
    features_path = args.holdout_features.resolve(strict=True)
    new_report_path = selected_run / "report.json"
    legacy_report_path = legacy_run / "report.json"
    for path in (new_report_path, legacy_report_path, holdout_dir / "fetch-report.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    selected_report = json.loads(new_report_path.read_text(encoding="utf-8"))
    legacy_report = json.loads(legacy_report_path.read_text(encoding="utf-8"))
    fetch_report = json.loads((holdout_dir / "fetch-report.json").read_text(encoding="utf-8"))
    if selected_report.get("status") != "complete_frozen_clevrer_binary_reasoning_readout":
        raise ValueError("new selected run is incomplete")
    if fetch_report.get("holdout_fetched_after_selection") is not True:
        raise ValueError("holdout was not fetched after checkpoint selection")
    if fetch_report.get("holdout_loaded_by_feature_extractor_or_trainer") is not False:
        raise ValueError("holdout was used before selected model evaluation")
    manifest_path = holdout_dir / "holdout.jsonl"
    manifest = _load_rows(manifest_path)
    if any(row.get("partition") != "holdout" for row in manifest):
        raise ValueError("holdout manifest has unexpected partition label")
    if fetch_report.get("historical_media_identity_overlap") != 0:
        raise ValueError("holdout media overlaps a historical artifact")
    if any(fetch_report.get("cross_partition_media_hash_overlap", {}).values()):
        raise ValueError("holdout media overlaps train/development media")

    source = config["source"]
    holdout_rows = []
    for split, key in (("train", "train_questions"), ("validation", "validation_questions")):
        split_manifest = [row for row in manifest if row["split"] == split]
        if not split_manifest:
            continue
        raw_path = Path(source[key]).resolve(strict=True)
        raw_scenes = json.loads(raw_path.read_text(encoding="utf-8"))
        holdout_rows.extend(
            expand_binary_reasoning_choices(raw_scenes, split_manifest, split=split)
        )
    if not holdout_rows:
        raise ValueError("no reasoning decisions expanded for holdout")
    if len({int(row["scene_index"]) for row in holdout_rows}) != len(
        {int(row["scene_index"]) for row in manifest}
    ):
        raise ValueError("holdout expansion lost one or more video scenes")

    with np.load(features_path, allow_pickle=False) as cache:
        hashes = cache["media_sha256"].astype(str)
        values = cache["features"].astype(np.float32)
    video_by_hash = {key: values[index] for index, key in enumerate(hashes)}
    expected_media = {str(row["media_sha256"]) for row in manifest}
    if set(video_by_hash) != expected_media or values.shape != (len(expected_media), 768):
        raise ValueError("holdout VideoPrism feature cache does not match holdout media")
    feature_report_path = features_path.with_name("cache-report.json")
    if not feature_report_path.is_file():
        raise FileNotFoundError("holdout feature-cache provenance report is missing")
    feature_report = json.loads(feature_report_path.read_text(encoding="utf-8"))
    if feature_report.get("status") != "complete_videoprism_frozen_feature_cache":
        raise ValueError("unexpected frozen feature cache status")
    if feature_report.get("holdout_only") is not True or feature_report.get(
        "holdout_fetched_after_selection"
    ) is not True:
        raise ValueError("holdout feature extraction was not isolated until after selection")
    if feature_report.get("checkpoint_sha256") != config["features"]["checkpoint_sha256"]:
        raise ValueError("holdout feature cache uses an unexpected VideoPrism checkpoint")
    if feature_report.get("unique_video_count") != len(expected_media):
        raise ValueError("feature cache unique video count mismatch")

    if not torch.cuda.is_available():
        raise RuntimeError("fixed VideoPrism/MiniLM evaluation requires local CUDA")
    device = torch.device("cuda")
    text_path = Path(config["text_encoder"]["model_path"]).resolve(strict=True)
    tokenizer = AutoTokenizer.from_pretrained(text_path, local_files_only=True)
    encoder = AutoModel.from_pretrained(text_path, local_files_only=True).to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    hidden_size = int(config["decision"]["hidden_size"])
    selected_checkpoint = selected_run / "best-readout.pt"
    legacy_checkpoint = legacy_run / "best-readout.pt"
    if sha256_file(selected_checkpoint) != selected_report["artifact_hashes"]["best-readout.pt"]:
        raise ValueError("selected readout hash mismatch")
    if sha256_file(legacy_checkpoint) != legacy_report["artifact_hashes"]["best-readout.pt"]:
        raise ValueError("legacy readout hash mismatch")

    targets, selected_probabilities, selected_metrics = _evaluate_readout(
        selected_checkpoint, holdout_rows, video_by_hash, tokenizer, encoder, device, hidden_size
    )
    legacy_targets, legacy_probabilities, legacy_metrics = _evaluate_readout(
        legacy_checkpoint, holdout_rows, video_by_hash, tokenizer, encoder, device, hidden_size
    )
    if targets != legacy_targets:
        raise ValueError("paired candidates disagree about target ordering")
    bootstrap = paired_scene_bootstrap(
        holdout_rows,
        targets,
        legacy_probabilities,
        selected_probabilities,
        replicates=args.bootstrap_replicates,
        seed=17,
    )
    predictions = []
    for index, (row, target) in enumerate(zip(holdout_rows, targets, strict=True)):
        chosen = int(selected_probabilities[index].argmax())
        predictions.append(
            {
                "id": row["id"],
                "scene_index": row["scene_index"],
                "parent_question_id": row["parent_question_id"],
                "question_type": row["question_type"],
                "target_index": target,
                "prediction_index": chosen,
                "target": row["options"][target],
                "prediction": row["options"][chosen],
                "reference_probabilities": legacy_probabilities[index].tolist(),
                "candidate_probabilities": selected_probabilities[index].tolist(),
            }
        )
    output.mkdir(parents=True, exist_ok=False)
    _write_jsonl(output / "paired-predictions.jsonl", predictions)
    _write_json(
        output / "report.json",
        {
            "status": "complete_one_time_fresh_clevrer_holdout_evaluation",
            "selection_preceded_holdout_fetch": True,
            "holdout_rows_sha256": sha256_file(manifest_path),
            "feature_cache_sha256": sha256_file(features_path),
            "selected_report_sha256": sha256_file(new_report_path),
            "selected_checkpoint_sha256": sha256_file(selected_checkpoint),
            "legacy_checkpoint_sha256": sha256_file(legacy_checkpoint),
            "validation_only_selection": True,
            "sealed_audit_loaded": False,
            "rows": len(holdout_rows),
            "scenes": len({int(row["scene_index"]) for row in holdout_rows}),
            "reference": {"id": legacy_report["experiment_id"], "metrics": legacy_metrics},
            "candidate": {"id": selected_report["experiment_id"], "metrics": selected_metrics},
            "paired_scene_bootstrap": bootstrap,
            "predictions_path": "paired-predictions.jsonl",
        },
    )


if __name__ == "__main__":
    main()


def paired_scene_bootstrap(
    rows: list[dict[str, Any]],
    targets: list[int] | np.ndarray,
    reference_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    *,
    replicates: int = 2000,
    seed: int = 17,
) -> dict[str, Any]:
    """Bootstrap paired metric differences, resampling whole video scenes."""
    targets_array = np.asarray(targets, dtype=np.int64)
    reference = np.asarray(reference_probabilities, dtype=np.float64)
    candidate = np.asarray(candidate_probabilities, dtype=np.float64)
    if len(rows) != len(targets_array) or reference.shape != candidate.shape:
        raise ValueError("row, target, and paired probability lengths/shapes must match")
    if reference.shape[0] != len(rows) or reference.ndim != 2:
        raise ValueError("probabilities must be [examples, classes]")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("probabilities must be finite")
    if replicates < 1:
        raise ValueError("replicates must be positive")

    scene_indices: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        kind = str(row["question_type"])
        if kind not in QUESTION_TYPES:
            raise ValueError(f"unknown question type: {kind}")
        scene_indices[int(row["scene_index"])].append(index)
    present_types = {str(row["question_type"]) for row in rows}
    if not set(QUESTION_TYPES).issubset(present_types):
        raise ValueError("bootstrap requires all three question types")

    rng = np.random.default_rng(seed)
    metric_names = ("accuracy", "nll", "brier", "ece_15_bins")
    deltas: dict[str, list[float]] = {key: [] for key in metric_names}
    scenes = np.asarray(sorted(scene_indices), dtype=np.int64)
    attempts = 0
    while len(deltas["accuracy"]) < replicates:
        attempts += 1
        if attempts > replicates * 100:
            raise ValueError("could not draw scene bootstrap samples containing every task type")
        sampled = rng.choice(scenes, size=len(scenes), replace=True)
        by_type: dict[str, dict[str, float]] = {}
        for kind in QUESTION_TYPES:
            indexes = np.asarray(
                [
                    index
                    for scene in sampled
                    for index in scene_indices[int(scene)]
                    if rows[index]["question_type"] == kind
                ],
                dtype=np.int64,
            )
            if not len(indexes):
                break
            by_type[kind] = {
                name: _metrics(candidate[indexes], targets_array[indexes])[name]
                - _metrics(reference[indexes], targets_array[indexes])[name]
                for name in metric_names
            }
        if len(by_type) != len(QUESTION_TYPES):
            continue
        for name in metric_names:
            deltas[name].append(
                float(np.mean([by_type[kind][name] for kind in QUESTION_TYPES]))
            )

    macro_delta: dict[str, float] = {}
    for name in metric_names:
        values = []
        for kind in QUESTION_TYPES:
            indexes = np.asarray(
                [i for i, row in enumerate(rows) if row["question_type"] == kind],
                dtype=np.int64,
            )
            values.append(
                _metrics(candidate[indexes], targets_array[indexes])[name]
                - _metrics(reference[indexes], targets_array[indexes])[name]
            )
        macro_delta[name] = float(np.mean(values))
    return {
        "independent_scenes": len({int(row["scene_index"]) for row in rows}),
        "replicates": replicates,
        "seed": seed,
        "macro_question_type_difference": macro_delta,
        "difference_ci_95": {
            name: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
            for name, values in deltas.items()
        },
        "difference_definition": "candidate minus reference",
        "resampling_unit": "scene within question type; paired predictions",
    }
