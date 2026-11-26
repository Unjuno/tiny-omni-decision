"""Compare a fixed multimodal CLEVRER scorer with text-only and shuffled-video controls."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from scripts.train_frozen_clevrer_binary_reasoning import (
    QUESTION_TYPES,
    _decision_metrics,
    _sample_weights,
    _validate_inputs,
)
from scripts.train_frozen_text_probe import CandidateScorer
from scripts.train_frozen_video_probe import _combine, _predict, _text_features
from tiny_omni_decision.dataset import sha256_file


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--video-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _load_feature_map(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        hashes = data["media_sha256"].astype(str).tolist()
        features = np.asarray(data["features"], dtype=np.float32)
    if len(hashes) != len(set(hashes)) or features.shape != (len(hashes), 768):
        raise ValueError("VideoPrism cache must contain unique media hashes and 768-d features")
    if not np.isfinite(features).all():
        raise ValueError("VideoPrism cache contains non-finite features")
    return dict(zip(hashes, features, strict=True))


def _scene_derangement(scene_ids: list[int], seed: int) -> dict[int, int]:
    """Return a deterministic donor-scene mapping with no scene mapped to itself."""
    ordered = sorted(set(scene_ids))
    if len(ordered) < 2:
        raise ValueError("video-shuffle control requires at least two unique scenes")
    rng = np.random.default_rng(seed)
    # A cyclic shift of the same random ordering is a guaranteed derangement.
    randomized = np.asarray(ordered, dtype=np.int64)[rng.permutation(len(ordered))]
    shuffled = np.roll(randomized, 1)
    mapping = dict(zip(randomized.tolist(), shuffled.tolist(), strict=True))
    if any(scene == donor for scene, donor in mapping.items()):
        raise AssertionError("cyclic video assignment unexpectedly contains a fixed point")
    return mapping


def _prediction_rows(
    rows: list[dict[str, Any]], logits: list[torch.Tensor], targets: list[int], model: str
) -> list[dict[str, Any]]:
    predictions = []
    for row, values, target in zip(rows, logits, targets, strict=True):
        probabilities = torch.softmax(values.float(), dim=-1).cpu().tolist()
        prediction = int(np.argmax(probabilities))
        predictions.append(
            {
                "id": row["id"],
                "scene_index": int(row["scene_index"]),
                "parent_question_id": row["parent_question_id"],
                "question_type": row["question_type"],
                "target_index": target,
                "prediction_index": prediction,
                "probabilities": probabilities,
                "model": model,
            }
        )
    return predictions


def _paired_scene_bootstrap(
    predictions: dict[str, list[dict[str, Any]]],
    rows: list[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Paired percentile CIs for macro accuracy/NLL/Brier over sampled scenes."""
    if replicates < 100:
        raise ValueError("use at least 100 paired bootstrap replicates")
    scene_ids = sorted({int(row["scene_index"]) for row in rows})
    if len(scene_ids) < 2:
        raise ValueError("paired scene bootstrap requires at least two scenes")
    by_scene_type: dict[int, dict[str, list[int]]] = {
        scene: {kind: [] for kind in QUESTION_TYPES} for scene in scene_ids
    }
    for index, row in enumerate(rows):
        by_scene_type[int(row["scene_index"])][str(row["question_type"])].append(index)

    # Store sufficient statistics per scene, task type, and model so each
    # bootstrap draw remains paired while avoiding per-example rescoring.
    stats: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for model, model_rows in predictions.items():
        if len(model_rows) != len(rows):
            raise ValueError(f"prediction count mismatch for {model}")
        model_stats: dict[int, dict[str, np.ndarray]] = {}
        for scene in scene_ids:
            model_stats[scene] = {}
            for kind in QUESTION_TYPES:
                indexes = by_scene_type[scene][kind]
                values = np.zeros(3, dtype=np.float64)
                for index in indexes:
                    item = model_rows[index]
                    probs = np.clip(np.asarray(item["probabilities"], dtype=np.float64), 1e-12, 1)
                    target = int(item["target_index"])
                    values += [float(int(np.argmax(probs)) == target), -np.log(probs[target]),
                               float(((probs - np.eye(2)[target]) ** 2).sum())]
                model_stats[scene][kind] = np.r_[values, len(indexes)]
        stats[model] = model_stats

    rng = np.random.default_rng(seed)
    samples: dict[str, dict[str, list[float]]] = {
        model: {metric: [] for metric in ("accuracy", "nll", "brier")}
        for model in predictions
    }
    deltas: dict[str, dict[str, list[float]]] = {}
    model_names = list(predictions)
    for model in model_names[1:]:
        deltas[f"{model}_minus_{model_names[0]}"] = {
            metric: [] for metric in ("accuracy", "nll", "brier")
        }
    for _ in range(replicates):
        selected = rng.choice(scene_ids, size=len(scene_ids), replace=True)
        current: dict[str, dict[str, float]] = {}
        for model in model_names:
            totals = {metric: 0.0 for metric in ("accuracy", "nll", "brier")}
            for kind in QUESTION_TYPES:
                aggregate = sum((stats[model][int(scene)][kind] for scene in selected),
                                start=np.zeros(4, dtype=np.float64))
                count = aggregate[3]
                if count == 0:
                    continue
                for metric, index in (("accuracy", 0), ("nll", 1), ("brier", 2)):
                    totals[metric] += aggregate[index] / count / len(QUESTION_TYPES)
            current[model] = totals
            for metric in totals:
                samples[model][metric].append(totals[metric])
        for model in model_names[1:]:
            key = f"{model}_minus_{model_names[0]}"
            for metric in current[model]:
                deltas[key][metric].append(current[model][metric] - current[model_names[0]][metric])

    def quantiles(values: list[float]) -> list[float]:
        return [float(value) for value in np.quantile(values, [0.025, 0.975])]

    return {
        "method": "paired scene bootstrap; resample 200 validation scene IDs with replacement",
        "replicates": replicates,
        "seed": seed,
        "independent_scenes": len(scene_ids),
        "interval": "95% percentile",
        "delta_intervals": {
            key: {metric: quantiles(values) for metric, values in metrics.items()}
            for key, metrics in deltas.items()
        },
    }


def _train_text_only(
    train_features: list[torch.Tensor],
    train_targets: list[int],
    train_rows: list[dict[str, Any]],
    *,
    device: torch.device,
) -> tuple[CandidateScorer, list[dict[str, float]]]:
    torch.manual_seed(17)
    scorer = CandidateScorer(train_features[0].shape[-1], hidden_size=128).to(device)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=1e-3, weight_decay=1e-3)
    weights = _sample_weights(train_rows)
    generator = torch.Generator().manual_seed(17)
    history = []
    batch_size = 64
    for epoch in range(1, 25):
        scorer.train()
        order = torch.randperm(len(train_features), generator=generator).tolist()
        losses = []
        for start in range(0, len(order), batch_size):
            indexes = order[start : start + batch_size]
            batch = torch.cat([train_features[index] for index in indexes]).to(device)
            scores = scorer(batch).squeeze(-1).reshape(len(indexes), 2)
            targets = torch.tensor([train_targets[index] for index in indexes], device=device)
            per_example = F.cross_entropy(scores, targets, reduction="none")
            loss = (per_example * weights[indexes].to(device)).sum() * (
                len(train_features) / len(indexes)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "training_loss": float(np.mean(losses))})
    return scorer.eval(), history


def main() -> None:
    args = _args()
    started = datetime.now(UTC).isoformat()
    total_started = time.perf_counter()
    config_path = args.config.resolve(strict=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checked = _validate_inputs(config_path, config)
    sample_dir = args.sample_dir.resolve(strict=True)
    new_rows = _read_jsonl(sample_dir / "validation.jsonl")
    fetch_report = json.loads((sample_dir / "fetch-report.json").read_text(encoding="utf-8"))
    if (
        fetch_report.get("status") != "complete_clevrer_binary_control_sample"
        or fetch_report.get("validation_scene_overlap_with_prior") != 0
        or fetch_report.get("validation_content_overlap_with_prior") != 0
        or fetch_report.get("sealed_audit_loaded") is not False
    ):
        raise ValueError("fresh CLEVRER sample provenance did not pass overlap/audit checks")
    if not new_rows or {row["question_type"] for row in new_rows} != set(QUESTION_TYPES):
        raise ValueError("fresh validation must contain all three CLEVRER reasoning types")
    if any(row.get("split") != "validation" for row in new_rows):
        raise ValueError("fresh validation file contains a non-validation split row")
    if len({row["id"] for row in new_rows}) != len(new_rows):
        raise ValueError("fresh validation contains duplicate IDs")
    train_rows = checked["train_rows"]
    train_ids = {row["id"] for row in train_rows}
    if train_ids & {row["id"] for row in new_rows}:
        raise ValueError("training and fresh validation IDs overlap")

    video_features_path = args.video_features.resolve(strict=True)
    new_video_by_hash = _load_feature_map(video_features_path)
    feature_report_path = video_features_path.with_name("cache-report.json")
    feature_report = json.loads(feature_report_path.read_text(encoding="utf-8"))
    required_new = {str(row["media_sha256"]) for row in new_rows}
    witness_rows = _read_jsonl(sample_dir / "train.jsonl")
    witness_hashes = {str(row["media_sha256"]) for row in witness_rows}
    if len(witness_rows) != 1 or witness_hashes & required_new:
        raise ValueError("feature-extractor witness must be one separate, non-validation asset")
    if set(new_video_by_hash) != required_new | witness_hashes:
        raise ValueError("fresh VideoPrism features do not match validation plus witness media")
    expected_feature_manifests = {
        "train": sha256_file(sample_dir / "train.jsonl"),
        "validation": sha256_file(sample_dir / "validation.jsonl"),
        "fetch_report": sha256_file(sample_dir / "fetch-report.json"),
    }
    if (
        feature_report.get("status") != "complete_videoprism_frozen_feature_cache"
        or feature_report.get("checkpoint_sha256")
        != "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
        or feature_report.get("dataset_revision")
        != "MIT-IBM/CLEVRER@98b842082ba4f7c18b6b9e3f39145871782a65ef"
        or feature_report.get("sample_manifest_sha256") != expected_feature_manifests
        or feature_report.get("feature_cache_sha256") != sha256_file(video_features_path)
    ):
        raise ValueError("fresh VideoPrism feature-cache provenance mismatch")
    del checked["feature_by_hash"]
    fresh_video = torch.from_numpy(
        np.stack([new_video_by_hash[str(row["media_sha256"])] for row in new_rows]).copy()
    )
    del new_video_by_hash

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(config["text_encoder"]["model_path"],
                                               local_files_only=True)
    text_encoder = AutoModel.from_pretrained(config["text_encoder"]["model_path"],
                                             local_files_only=True).eval().to(device)
    for parameter in text_encoder.parameters():
        parameter.requires_grad_(False)
    text_encoding_started = time.perf_counter()
    train_text, train_targets = _text_features(train_rows, tokenizer, text_encoder, device)
    fresh_text, fresh_targets = _text_features(new_rows, tokenizer, text_encoder, device)
    text_encoding_seconds = time.perf_counter() - text_encoding_started
    del text_encoder, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    fresh_multimodal = _combine(fresh_video, fresh_text)
    train_text_only = train_text
    old_model_path = Path(config["outputs"]["run_dir"]) / "best-readout.pt"
    old_state = torch.load(old_model_path, map_location="cpu", weights_only=True)
    multimodal = CandidateScorer(2304, hidden_size=128).to(device)
    multimodal.load_state_dict(old_state, strict=True)
    multimodal.eval()
    scorer_eval_started = time.perf_counter()
    multimodal_logits = _predict(multimodal, fresh_multimodal, device)
    fixed_model_eval_seconds = time.perf_counter() - scorer_eval_started
    text_training_started = time.perf_counter()
    text_scorer, training_history = _train_text_only(
        train_text_only, train_targets, train_rows, device=device
    )
    text_training_seconds = time.perf_counter() - text_training_started
    text_eval_started = time.perf_counter()
    text_logits = _predict(text_scorer, fresh_text, device)
    text_eval_seconds = time.perf_counter() - text_eval_started

    mapping = _scene_derangement([int(row["scene_index"]) for row in new_rows], 17)
    unique_video = {
        int(row["scene_index"]): fresh_video[index]
        for index, row in enumerate(new_rows)
    }
    shuffled_features = []
    for index, row in enumerate(new_rows):
        donor = mapping[int(row["scene_index"])]
        options = fresh_text[index]
        video = unique_video[donor].expand(len(options), -1)
        shuffled_features.append(torch.cat([video, options], dim=-1))
    shuffle_eval_started = time.perf_counter()
    shuffled_logits = _predict(multimodal, shuffled_features, device)
    shuffle_eval_seconds = time.perf_counter() - shuffle_eval_started

    prediction_sets = {
        "videoprism_plus_text_fixed_v0": _prediction_rows(
            new_rows, multimodal_logits, fresh_targets, "videoprism_plus_text_fixed_v0"
        ),
        "text_only_seed17_epoch24": _prediction_rows(
            new_rows, text_logits, fresh_targets, "text_only_seed17_epoch24"
        ),
        "video_shuffled_scene_derangement": _prediction_rows(
            new_rows, shuffled_logits, fresh_targets, "video_shuffled_scene_derangement"
        ),
    }
    metrics = {
        name: _decision_metrics(logits, fresh_targets, new_rows)
        for name, logits in (
            ("videoprism_plus_text_fixed_v0", multimodal_logits),
            ("text_only_seed17_epoch24", text_logits),
            ("video_shuffled_scene_derangement", shuffled_logits),
        )
    }
    bootstrap_started = time.perf_counter()
    bootstrap = _paired_scene_bootstrap(
        prediction_sets,
        new_rows,
        replicates=args.bootstrap_replicates,
        seed=17,
    )
    bootstrap_seconds = time.perf_counter() - bootstrap_started

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    output.mkdir(parents=True)
    prediction_path = output / "validation-predictions.jsonl"
    all_predictions = [
        {**prediction, "id": row["id"]}
        for model_rows in prediction_sets.values()
        for prediction, row in zip(model_rows, new_rows, strict=True)
    ]
    prediction_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_predictions),
        encoding="utf-8",
        newline="\n",
    )
    torch.save(text_scorer.state_dict(), output / "text-only-readout.pt")
    (output / "text-only-training-history.json").write_text(
        json.dumps(training_history, indent=2) + "\n", encoding="utf-8"
    )
    evaluation_config = {
        "seed": 17,
        "validation_manifest": str(sample_dir / "validation.jsonl"),
        "video_feature_cache": str(video_features_path),
        "fixed_multimodal_checkpoint": str(old_model_path),
        "text_only_training": {
            "epochs": 24,
            "batch_examples": 64,
            "learning_rate": 0.001,
            "weight_decay": 0.001,
            "optimizer": "AdamW",
            "checkpoint_rule": "fixed final epoch; no validation selection",
        },
        "video_shuffle": "seeded scene-level derangement; donor video shared within scene",
        "paired_bootstrap": {"replicates": args.bootstrap_replicates, "seed": 17},
    }
    config_artifact = output / "evaluation-config.json"
    config_artifact.write_text(json.dumps(evaluation_config, indent=2) + "\n", encoding="utf-8")
    report = {
        "schema_version": 1,
        "status": "complete_clevrer_video_text_control_evaluation",
        "claim_scope": (
            "fixed, non-blind development validation on previously unused CLEVRER scenes; "
            "not a sealed audit or general video understanding claim"
        ),
        "run_started_utc": started,
        "run_finished_utc": datetime.now(UTC).isoformat(),
        "seed": 17,
        "sealed_audit_loaded": False,
        "checkpoint_selection_on_fresh_validation": False,
        "training": {
            "text_only_train_rows": len(train_rows),
            "unique_train_scenes": len({row["scene_index"] for row in train_rows}),
            "unique_train_assets": len({row["media_sha256"] for row in train_rows}),
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.001,
            "epochs": 24,
            "batch_examples": 64,
            "sampling_weights": "equal question type, then equal scene within type",
            "checkpoint": "fixed epoch 24; fresh validation was not consulted during training",
        },
        "validation": {
            "source_revision": "MIT-IBM/CLEVRER@98b842082ba4f7c18b6b9e3f39145871782a65ef",
            "sample_dir": str(sample_dir),
            "sample_hashes": {
                "validation_jsonl": sha256_file(sample_dir / "validation.jsonl"),
                "fetch_report": sha256_file(sample_dir / "fetch-report.json"),
                "video_feature_cache": sha256_file(video_features_path),
                "video_feature_cache_report": sha256_file(feature_report_path),
            },
            "row_count": len(new_rows),
            "scene_count": len({row["scene_index"] for row in new_rows}),
            "unique_video_assets": len({row["media_sha256"] for row in new_rows}),
            "parent_question_count": len({row["parent_question_id"] for row in new_rows}),
            "question_type_counts": dict(Counter(row["question_type"] for row in new_rows)),
            "scene_video_shuffle_derangement": {str(k): int(v) for k, v in mapping.items()},
        },
        "models": {
            "videoprism_plus_text_fixed_v0": {
                "checkpoint": str(old_model_path),
                "checkpoint_sha256": sha256_file(old_model_path),
                "readout_not_retrained": True,
            },
            "text_only_seed17_epoch24": {
                "selection_rule": "fixed final epoch, no validation selection",
                "trainable_parameters": sum(p.numel() for p in text_scorer.parameters()),
            },
            "video_shuffled_scene_derangement": {
                "checkpoint": str(old_model_path),
                "video_assignment_seed": 17,
                "same_donor_video_for_all_questions_in_scene": True,
            },
        },
        "metrics": metrics,
        "paired_scene_bootstrap": bootstrap,
        "timing_seconds": {
            "minilm_feature_encoding_train_and_validation": text_encoding_seconds,
            "fixed_multimodal_scorer_evaluation": fixed_model_eval_seconds,
            "text_only_readout_training_24_epochs": text_training_seconds,
            "text_only_scorer_evaluation": text_eval_seconds,
            "video_shuffled_scorer_evaluation": shuffle_eval_seconds,
            "paired_scene_bootstrap": bootstrap_seconds,
            "total_including_provenance_and_output": time.perf_counter() - total_started,
        },
        "provenance": {
            "source_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
            ).stdout.strip(),
            "evaluation_config_sha256": sha256_file(config_artifact),
            "evaluator_script_sha256": sha256_file(Path(__file__).resolve()),
            "training_script_sha256": sha256_file(
                Path("scripts/train_frozen_clevrer_binary_reasoning.py").resolve()
            ),
            "source_training_config_sha256": checked["config_sha256"],
            "source_training_report_sha256": sha256_file(
                Path(config["outputs"]["run_dir"]).resolve() / "report.json"
            ),
        },
        "environment": {
            "torch": torch.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda" else None,
        },
        "artifact_hashes": {
            "validation_predictions": sha256_file(prediction_path),
            "text_only_readout": sha256_file(output / "text-only-readout.pt"),
            "text_only_training_history": sha256_file(output / "text-only-training-history.json"),
            "evaluation_config": sha256_file(config_artifact),
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "metrics": metrics,
                      "paired_scene_bootstrap": bootstrap}, indent=2))


if __name__ == "__main__":
    main()
