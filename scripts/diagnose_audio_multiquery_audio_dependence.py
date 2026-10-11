"""Measure frozen audio-feature contribution to a saved multi-query readout."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from scripts.train_frozen_audio_multiquery import (
    QUESTION_TASKS,
    CandidateScorer,
    _encode_texts,
    _evaluate,
    build_query_specs,
    make_query_examples,
)
from scripts.train_frozen_audio_probe import LABELS, _rows, _validate_splits
from scripts.verify_audio_multiquery_observation_cache import (
    _compare_predictions,
    _tensor_from_payload,
)
from tiny_omni_decision.cache import ObservationFeatureCache
from tiny_omni_decision.dataset import sha256_file
from tiny_omni_decision.observation_identity import audio_observation_feature_key


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/pretrained_reuse/path_a_audio_multiquery.yaml")
    )
    parser.add_argument(
        "--run-dir", type=Path, default=Path("C:/CodexArtifacts/pretrained-reuse-r2/audio-multiquery-v0")
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("C:/CodexArtifacts/pretrained-reuse-r2/audio-observation-cache-multiquery-v4"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("C:/CodexArtifacts/pretrained-reuse-r2/audio-multiquery-audio-dependence-v0"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def _ordered_ids_sha256(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256("\n".join(str(row["id"]) for row in rows).encode()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _text_payloads() -> set[str]:
    return {
        f"Question: {spec['question']}"
        for label in LABELS
        for spec in build_query_specs(label)
    } | {
        f"Candidate answer: {option}."
        for label in LABELS
        for spec in build_query_specs(label)
        for option in spec["options"]
    }


def _derangement(size: int, seed: int) -> Tensor:
    if size < 2:
        raise ValueError("at least two observations are required to shuffle audio features")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(10_000):
        order = torch.randperm(size, generator=generator)
        if torch.all(order != torch.arange(size)):
            return order
    raise RuntimeError("could not construct a non-identity audio permutation")


def _prediction_tensors(
    rows: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> dict[str, dict[str, Tensor]]:
    expected = len(rows) * len(QUESTION_TASKS)
    if len(predictions) != expected:
        raise ValueError(f"expected {expected} query predictions, found {len(predictions)}")
    by_key = {(str(item["sample_id"]), str(item["question_type"])): item for item in predictions}
    if len(by_key) != len(predictions):
        raise ValueError("duplicate sample/query prediction identity")
    result: dict[str, dict[str, Tensor]] = {}
    for task in QUESTION_TASKS:
        ordered = []
        for row in rows:
            key = (str(row["id"]), task)
            if key not in by_key:
                raise ValueError(f"missing prediction for {key}")
            ordered.append(by_key[key])
        option_count = len(ordered[0]["options"])
        if any(len(item["options"]) != option_count for item in ordered):
            raise ValueError(f"option count changed within question type {task}")
        result[task] = {
            "logits": torch.tensor([item["logits"] for item in ordered], dtype=torch.float64),
            "targets": torch.tensor([item["target_index"] for item in ordered], dtype=torch.long),
        }
    return result


def _metric_values(logits: Tensor, targets: Tensor) -> dict[str, float]:
    probabilities = torch.softmax(logits, dim=-1)
    confidence, predicted = probabilities.max(dim=-1)
    correct = predicted.eq(targets).to(torch.float64)
    nll = -probabilities.gather(1, targets[:, None]).squeeze(1).clamp_min(1e-12).log()
    one_hot = torch.nn.functional.one_hot(targets, num_classes=logits.shape[-1]).to(torch.float64)
    brier = (probabilities - one_hot).square().sum(dim=-1)
    ece = torch.zeros((), dtype=torch.float64)
    for index in range(15):
        low, high = index / 15, (index + 1) / 15
        selected = (confidence >= low) & (
            confidence < high if index < 14 else confidence <= high
        )
        if selected.any():
            ece += selected.sum() / len(targets) * (
                correct[selected].mean() - confidence[selected].mean()
            ).abs()
    return {
        "accuracy": float(correct.mean()),
        "nll": float(nll.mean()),
        "brier": float(brier.mean()),
        "ece_15_bins": float(ece),
    }


def _macro_metrics(by_task: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        key: sum(by_task[task][key] for task in QUESTION_TASKS) / len(QUESTION_TASKS)
        for key in ("accuracy", "nll", "brier", "ece_15_bins")
    }


def _group_bootstrap_indices(
    rows: list[dict[str, Any]], replicates: int, seed: int
) -> list[list[int]]:
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    speaker_rows: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        speaker_rows[str(row["speaker_group_sha256"])].append(index)
    speakers = sorted(speaker_rows)
    generator = random.Random(seed)
    result = []
    for _ in range(replicates):
        sampled_speakers = [generator.choice(speakers) for _ in speakers]
        result.append(
            [
                index
                for speaker in sampled_speakers
                for index in speaker_rows[speaker]
            ]
        )
    return result


def _bootstrap_differences(
    rows: list[dict[str, Any]],
    reference: dict[str, dict[str, Tensor]],
    candidates: dict[str, dict[str, Tensor]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    draws = _group_bootstrap_indices(rows, replicates, seed)
    values: dict[str, dict[str, list[float]]] = {
        scope: {metric: [] for metric in ("accuracy", "nll", "brier", "ece_15_bins")}
        for scope in ("macro_question_type", *QUESTION_TASKS)
    }
    for indices in draws:
        ref_by_task = {}
        candidate_by_task = {}
        for task in QUESTION_TASKS:
            index = torch.tensor(indices, dtype=torch.long)
            ref_by_task[task] = _metric_values(
                reference[task]["logits"][index], reference[task]["targets"][index]
            )
            candidate_by_task[task] = _metric_values(
                candidates[task]["logits"][index], candidates[task]["targets"][index]
            )
        for scope, ref_metrics, candidate_metrics in [
            ("macro_question_type", _macro_metrics(ref_by_task), _macro_metrics(candidate_by_task)),
            *[
                (task, ref_by_task[task], candidate_by_task[task])
                for task in QUESTION_TASKS
            ],
        ]:
            for metric in values[scope]:
                values[scope][metric].append(candidate_metrics[metric] - ref_metrics[metric])
    intervals = {}
    for scope, metrics in values.items():
        intervals[scope] = {}
        for metric, samples in metrics.items():
            ordered = sorted(samples)
            intervals[scope][metric] = {
                "low_95_percentile": ordered[int(0.025 * (replicates - 1))],
                "high_95_percentile": ordered[int(0.975 * (replicates - 1))],
                "mean_difference": sum(samples) / replicates,
            }
    return {
        "unit": "speaker_group_sha256 cluster; all clips and four queries retained per sampled speaker",
        "speaker_clusters": len({str(row["speaker_group_sha256"]) for row in rows}),
        "replicates": replicates,
        "seed": seed,
        "interval_method": "paired cluster bootstrap; percentile 95% interval",
        "difference_sign": "condition minus original audio; negative NLL/Brier is better",
        "by_scope": intervals,
    }


def main() -> None:
    args = _args()
    started = datetime.now(UTC)
    started_clock = time.perf_counter()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = args.run_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audio-dependence output: {output}")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)

    report_path = run_dir / "run-report.json"
    expected_predictions_path = run_dir / "validation-multiquery-predictions.jsonl"
    checkpoint_path = run_dir / "best-readout.pt"
    effective_config_path = run_dir / "effective-config.yaml"
    source_cache_path = Path(config["source"]["feature_cache"]).resolve()
    cache_report_path = cache_dir / "verification-report.json"
    cache_manifest_path = cache_dir / "cache-records.jsonl"
    required = (
        config_path,
        report_path,
        expected_predictions_path,
        checkpoint_path,
        effective_config_path,
        source_cache_path,
        cache_report_path,
        cache_manifest_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required frozen inputs are missing: {missing}")

    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    cache_report = json.loads(cache_report_path.read_text(encoding="utf-8"))
    if report.get("status") != "complete_frozen_audio_multiquery_development_experiment":
        raise ValueError("source run is not a completed frozen multi-query experiment")
    if report.get("sealed_audit_loaded") is not False or report.get("test_split_loaded") is not False:
        raise ValueError("source run indicates forbidden test or sealed-audit use")
    expected_hashes = {
        config_path: report["config_sha256"],
        effective_config_path: report["effective_config_sha256"],
        source_cache_path: report["data_hashes"]["feature_cache"],
        checkpoint_path: report["model"]["readout_state_dict_sha256"],
        expected_predictions_path: report["validation_predictions_sha256"],
    }
    for path, expected in expected_hashes.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen input hash mismatch for {path}: {actual} != {expected}")
    if sha256_file(report_path) != cache_report["multiquery_source_run_report_sha256"]:
        raise ValueError("observation-cache report does not match the source experiment report")
    if cache_report.get("status") != "complete_audio_observation_cache_validation":
        raise ValueError("audio feature cache has not passed its parity verifier")
    if sha256_file(cache_manifest_path) != cache_report["cache_manifest_sha256"]:
        raise ValueError("audio observation-cache manifest hash mismatch")
    if (
        cache_report["source_train_manifest_sha256"] != report["data_hashes"]["train_manifest"]
        or cache_report["source_validation_manifest_sha256"]
        != report["data_hashes"]["validation_manifest"]
        or cache_report["source_feature_cache_sha256"] != report["data_hashes"]["feature_cache"]
    ):
        raise ValueError("observation cache was built from different frozen experiment inputs")

    sample_dir = Path(config["source"]["sample_dir"]).resolve()
    train_path, validation_path = sample_dir / "train.jsonl", sample_dir / "validation.jsonl"
    train_rows, validation_rows = _rows(train_path), _rows(validation_path)
    _validate_splits(sample_dir, train_rows, validation_rows)
    if sha256_file(train_path) != report["data_hashes"]["train_manifest"]:
        raise ValueError("train manifest changed")
    if sha256_file(validation_path) != report["data_hashes"]["validation_manifest"]:
        raise ValueError("validation manifest changed")
    if _ordered_ids_sha256(validation_rows) != cache_report["validation_ordered_ids_sha256"]:
        raise ValueError("validation ID order differs from the verified observation cache")
    source_features = torch.load(source_cache_path, map_location="cpu", weights_only=True)
    if source_features.get("validation_ids") != [str(row["id"]) for row in validation_rows]:
        raise ValueError("frozen audio feature-cache validation ID order changed")
    original_audio = source_features["validation_features"].float().contiguous()
    if original_audio.shape != (len(validation_rows), 384) or not torch.isfinite(original_audio).all():
        raise ValueError("frozen validation audio feature matrix has an invalid shape or value")

    cache_rows = _jsonl(cache_manifest_path)
    val_cache_rows = [row for row in cache_rows if row["split"] == "validation"]
    if [row["sample_id"] for row in val_cache_rows] != [str(row["id"]) for row in validation_rows]:
        raise ValueError("observation cache records do not match frozen validation IDs in order")
    preprocessing = cache_report["preprocessing_spec"]
    observation_cache = ObservationFeatureCache(cache_dir / "feature-cache")
    cached_features = []
    media_bytes = 0
    for index, (row, manifest_row) in enumerate(zip(validation_rows, val_cache_rows, strict=True)):
        media_path = sample_dir / str(row["media_path"])
        if not media_path.is_file() or sha256_file(media_path) != row["media_sha256"]:
            raise ValueError(f"missing or changed validation audio asset: {row['id']}")
        media_bytes += media_path.stat().st_size
        key = audio_observation_feature_key(
            row,
            encoder_revision=str(config["encoder"]["audio_revision"]),
            encoder_weights_sha256=str(config["encoder"]["audio_weights_sha256"]),
            extraction_code_sha256=str(preprocessing["feature_extraction_code_sha256"]),
            preprocessor_config_sha256=str(preprocessing["preprocessor_config_sha256"]),
            preprocessing_sha256=str(cache_report["preprocessing_sha256"]),
        )
        if key.cache_id != manifest_row["cache_id"]:
            raise ValueError(f"audio cache identity mismatch for {row['id']}")
        cached = observation_cache.get(key)
        if cached.payload_sha256 != manifest_row["feature_payload_sha256"]:
            raise ValueError(f"cached audio feature hash mismatch for {row['id']}")
        feature = _tensor_from_payload(cached.payload, key.feature_shape)
        if not torch.equal(feature, original_audio[index]):
            raise ValueError(f"observation-cache feature differs from source cache for {row['id']}")
        cached_features.append(feature)
    audio = torch.stack(cached_features)
    if not torch.equal(audio, original_audio):
        raise ValueError("reloaded validation audio features differ from the frozen source matrix")

    train_speakers = {str(row["speaker_group_sha256"]) for row in train_rows}
    validation_speakers = {str(row["speaker_group_sha256"]) for row in validation_rows}
    train_media = {str(row["media_sha256"]) for row in train_rows}
    validation_media = {str(row["media_sha256"]) for row in validation_rows}
    if train_speakers & validation_speakers or train_media & validation_media:
        raise ValueError("frozen train and validation data have speaker or media overlap")

    text_vectors, text_parameter_count = _encode_texts(
        sorted(_text_payloads()), Path(config["encoder"]["text_model_path"]).resolve(), torch.device("cpu")
    )
    if text_parameter_count != int(config["encoder"]["text_parameters"]):
        raise ValueError("pinned MiniLM parameter count changed")
    input_examples = make_query_examples(validation_rows, audio, text_vectors)
    scorer = CandidateScorer(int(input_examples[0]["features"].shape[-1]))
    scorer.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    expected_predictions = _jsonl(expected_predictions_path)
    expected_ids = [(row["id"], task) for row in validation_rows for task in QUESTION_TASKS]
    actual_ids = [(item["sample_id"], item["question_type"]) for item in expected_predictions]
    if actual_ids != expected_ids:
        raise ValueError("saved baseline prediction identity/order differs from validation query order")

    permutation = _derangement(len(validation_rows), args.seed)
    shuffled_audio = audio[permutation]
    shuffled_source_same_speaker = sum(
        str(validation_rows[index]["speaker_group_sha256"])
        == str(validation_rows[int(permutation[index])]["speaker_group_sha256"])
        for index in range(len(validation_rows))
    )
    conditions: dict[str, tuple[Tensor, list[dict[str, Any]]]] = {}
    evaluation_seconds: dict[str, float] = {}
    for name, features in (
        ("original_audio", audio),
        ("zero_audio", torch.zeros_like(audio)),
        ("shuffled_other_clip_audio", shuffled_audio),
    ):
        examples = make_query_examples(validation_rows, features, text_vectors)
        started_eval = time.perf_counter()
        metrics, predictions = _evaluate(
            scorer, examples, torch.device("cpu"), int(config["readout"]["batch_queries"])
        )
        evaluation_seconds[name] = time.perf_counter() - started_eval
        conditions[name] = (features, predictions)
        if name == "original_audio":
            max_logit_delta, max_probability_delta = _compare_predictions(
                expected_predictions, predictions
            )
            for group in ("macro_question_type", *QUESTION_TASKS):
                for metric in ("accuracy", "nll", "brier", "ece_15_bins"):
                    actual_metric = (
                        metrics["macro_question_type"][metric]
                        if group == "macro_question_type"
                        else metrics["by_question_type"][group][metric]
                    )
                    expected_metric = (
                        report["validation_metrics"]["macro_question_type"][metric]
                        if group == "macro_question_type"
                        else report["validation_metrics"]["by_question_type"][group][metric]
                    )
                    delta = abs(float(actual_metric) - float(expected_metric))
                    if delta > 2e-5:
                        raise ValueError(f"baseline metric parity failed for {group}.{metric}: {delta}")
            if max_logit_delta > 2e-5 or max_probability_delta > 2e-6:
                raise ValueError(
                    f"saved baseline predictions did not reload within tolerance: {max_logit_delta}, {max_probability_delta}"
                )
    predictions_by_condition = {
        name: _prediction_tensors(validation_rows, predictions)
        for name, (_, predictions) in conditions.items()
    }

    metrics_by_condition: dict[str, Any] = {}
    for name, (_, predictions) in conditions.items():
        by_task = {}
        keyed = {(item["sample_id"], item["question_type"]): item for item in predictions}
        for task in QUESTION_TASKS:
            task_predictions = [keyed[(str(row["id"]), task)] for row in validation_rows]
            logits = [torch.tensor(item["logits"], dtype=torch.float32) for item in task_predictions]
            targets = [int(item["target_index"]) for item in task_predictions]
            from scripts.train_frozen_text_probe import _metrics

            by_task[task] = _metrics(logits, targets)
        metrics_by_condition[name] = {
            "by_question_type": by_task,
            "macro_question_type": _macro_metrics(
                {task: {key: float(by_task[task][key]) for key in ("accuracy", "nll", "brier", "ece_15_bins")}
                 for task in QUESTION_TASKS}
            ),
            "prediction_sha256": None,
        }

    bootstrap = {
        name: _bootstrap_differences(
            validation_rows,
            predictions_by_condition["original_audio"],
            predictions_by_condition[name],
            replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
        for name in ("zero_audio", "shuffled_other_clip_audio")
    }
    output.mkdir(parents=True, exist_ok=False)
    prediction_hashes = {}
    for name, (_, predictions) in conditions.items():
        path = output / f"{name}-predictions.jsonl"
        path.write_text(
            "".join(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n" for item in predictions),
            encoding="utf-8",
        )
        digest = sha256_file(path)
        prediction_hashes[name] = digest
        metrics_by_condition[name]["prediction_sha256"] = digest

    config_hash = sha256_file(config_path)
    run_hash = sha256_file(report_path)
    cache_report_hash = sha256_file(cache_report_path)
    cache_manifest_hash = sha256_file(cache_manifest_path)
    try:
        transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        transformers_version = "UNKNOWN"
    try:
        torch_version = torch.__version__
    except Exception:
        torch_version = "UNKNOWN"
    permutation_hash = hashlib.sha256(
        ",".join(str(int(index)) for index in permutation).encode()
    ).hexdigest()
    report_output = {
        "schema_version": 1,
        "status": "complete_posthoc_audio_feature_dependence_diagnostic",
        "diagnostic_only": True,
        "not_for_model_selection": True,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(UTC).isoformat(),
        "wall_seconds": time.perf_counter() - started_clock,
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "seed": args.seed,
        "device": "CPU",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": str(torch_version),
            "transformers": transformers_version,
        },
        "source_run": {
            "run_report_sha256": run_hash,
            "config_sha256": config_hash,
            "effective_config_sha256": sha256_file(effective_config_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "saved_baseline_predictions_sha256": sha256_file(expected_predictions_path),
            "validation_manifest_sha256": sha256_file(validation_path),
            "validation_ordered_ids_sha256": _ordered_ids_sha256(validation_rows),
            "source_audio_feature_cache_sha256": sha256_file(source_cache_path),
            "observation_cache_report_sha256": cache_report_hash,
            "observation_cache_manifest_sha256": cache_manifest_hash,
            "text_encoder_revision": config["encoder"]["text_revision"],
            "text_encoder_weights_sha256": config["encoder"]["text_weights_sha256"],
            "audio_encoder_revision": config["encoder"]["audio_revision"],
            "audio_encoder_weights_sha256": config["encoder"]["audio_weights_sha256"],
        },
        "validation": {
            "clip_count": len(validation_rows),
            "unique_audio_asset_count": len(validation_media),
            "speaker_cluster_count": len(validation_speakers),
            "train_validation_audio_asset_overlap": len(train_media & validation_media),
            "train_validation_speaker_overlap": len(train_speakers & validation_speakers),
            "validation_audio_bytes_verified": media_bytes,
            "queries_per_clip": len(QUESTION_TASKS),
            "query_count": len(expected_predictions),
            "question_types": list(QUESTION_TASKS),
            "all_four_queries_share_one_observation_feature": True,
        },
        "baseline_reload_parity": {
            "max_logit_delta": max_logit_delta,
            "max_probability_delta": max_probability_delta,
            "saved_prediction_order_matches_validation": True,
        },
        "conditions": {
            "original_audio": "Original Whisper feature for each clip; unchanged text/query construction.",
            "zero_audio": "Zero the 384D audio feature before rebuilding all audio-dependent pair terms.",
            "shuffled_other_clip_audio": (
                "Replace each clip feature with a different validation clip's Whisper feature; "
                "one seeded derangement is shared across all four query types."
            ),
        },
        "audio_shuffle": {
            "permutation_sha256": permutation_hash,
            "seed": args.seed,
            "fixed_points": int((permutation == torch.arange(len(validation_rows))).sum()),
            "same_speaker_source_matches": shuffled_source_same_speaker,
            "all_sources_are_different_clip_ids": True,
        },
        "metrics": metrics_by_condition,
        "paired_cluster_bootstrap": bootstrap,
        "evaluation_seconds": evaluation_seconds,
        "interpretation_limitations": [
            "The same development validation was used to select the saved readout; this post-hoc diagnostic is not a blind or independent evaluation.",
            "Speech Commands is a narrow closed-set keyword task; results do not establish general speech understanding or synchronized audio-video fusion.",
            "Zeroing or shuffling audio tests dependence of this fixed readout on the cached audio feature, not causal modality quality in a jointly trained model.",
            "Cluster bootstrap resamples validation speakers; with only the available speaker groups it is an uncertainty diagnostic, not a replacement for new held-out data.",
        ],
        "training_performed": False,
        "sealed_audit_loaded": False,
        "test_split_loaded": False,
        "output_prediction_sha256": prediction_hashes,
    }
    report_path_out = output / "diagnostic-report.json"
    report_path_out.write_text(
        json.dumps(report_output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if report_path.read_bytes() != report_bytes:
        raise RuntimeError("source run report changed during the read-only diagnostic")
    print(json.dumps(report_output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
