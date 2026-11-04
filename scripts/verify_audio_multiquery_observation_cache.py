"""Persist frozen Speech Commands observations and verify multi-query parity."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from scripts.train_frozen_audio_multiquery import (
    QUESTION_TASKS,
    CandidateScorer,
    _encode_texts,
    _evaluate,
    build_query_specs,
    make_query_examples,
)
from scripts.train_frozen_audio_probe import LABELS, _rows, _validate_splits
from tiny_omni_decision.cache import ObservationFeatureCache
from tiny_omni_decision.dataset import sha256_file
from tiny_omni_decision.observation_identity import audio_observation_feature_key


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/pretrained_reuse/path_a_audio_multiquery.yaml")
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _ordered_ids_sha256(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256("\n".join(str(row["id"]) for row in rows).encode()).hexdigest()


def _tensor_from_payload(payload: bytes, shape: tuple[int, ...]) -> torch.Tensor:
    expected_bytes = torch.empty(shape, dtype=torch.float32).numel() * 4
    if len(payload) != expected_bytes:
        raise ValueError(f"feature payload has {len(payload)} bytes; expected {expected_bytes}")
    return torch.frombuffer(bytearray(payload), dtype=torch.float32).reshape(shape).clone()


def _compare_predictions(
    expected: list[dict[str, Any]], actual: list[dict[str, Any]]
) -> tuple[float, float]:
    if len(expected) != len(actual):
        raise ValueError(f"prediction count changed: {len(expected)} != {len(actual)}")
    max_logit_delta = 0.0
    max_probability_delta = 0.0
    for saved, reloaded in zip(expected, actual, strict=True):
        for field in (
            "sample_id",
            "target_label",
            "question_type",
            "target_index",
            "prediction_index",
            "options",
        ):
            if saved[field] != reloaded[field]:
                raise ValueError(f"prediction identity differs in field {field}")
        for field, accumulator_name in (
            ("logits", "logit"),
            ("probabilities", "probability"),
        ):
            delta = max(
                abs(float(left) - float(right))
                for left, right in zip(saved[field], reloaded[field], strict=True)
            )
            if accumulator_name == "logit":
                max_logit_delta = max(max_logit_delta, delta)
            else:
                max_probability_delta = max(max_probability_delta, delta)
    return max_logit_delta, max_probability_delta


def main() -> None:
    args = _args()
    started = datetime.now(UTC)
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = args.run_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite observation-cache output: {output}")
    if not torch.cuda.is_available():
        device_name = "CPU"
    else:
        device_name = "CPU (explicitly selected; no GPU inference)"
    torch.set_num_threads(min(8, torch.get_num_threads()))

    report_path = run_dir / "run-report.json"
    expected_predictions_path = run_dir / "validation-multiquery-predictions.jsonl"
    checkpoint_path = run_dir / "best-readout.pt"
    effective_config_path = run_dir / "effective-config.yaml"
    source_cache_path = Path(config["source"]["feature_cache"]).resolve()
    sample_dir = Path(config["source"]["sample_dir"]).resolve()
    for path in (
        report_path,
        expected_predictions_path,
        checkpoint_path,
        effective_config_path,
        source_cache_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"required frozen input is missing: {path}")
    original_report = report_path.read_bytes()
    report = json.loads(original_report)
    if report.get("status") != "complete_frozen_audio_multiquery_development_experiment":
        raise ValueError("input report is not the expected completed audio run")
    if (
        report.get("sealed_audit_loaded") is not False
        or report.get("test_split_loaded") is not False
    ):
        raise ValueError("input run report indicates forbidden test or sealed-audit use")
    if sha256_file(config_path) != report["config_sha256"]:
        raise ValueError("source experiment config hash mismatch")
    if sha256_file(effective_config_path) != report["effective_config_sha256"]:
        raise ValueError("saved run config hash mismatch")
    if sha256_file(source_cache_path) != report["data_hashes"]["feature_cache"]:
        raise ValueError("frozen source feature-cache hash mismatch")
    if sha256_file(checkpoint_path) != report["model"]["readout_state_dict_sha256"]:
        raise ValueError("saved readout checkpoint hash mismatch")
    if sha256_file(expected_predictions_path) != report["validation_predictions_sha256"]:
        raise ValueError("saved validation-prediction hash mismatch")

    train_path = sample_dir / "train.jsonl"
    validation_path = sample_dir / "validation.jsonl"
    train_rows = _rows(train_path)
    validation_rows = _rows(validation_path)
    _validate_splits(sample_dir, train_rows, validation_rows)
    if sha256_file(train_path) != report["data_hashes"]["train_manifest"]:
        raise ValueError("frozen train manifest hash mismatch")
    if sha256_file(validation_path) != report["data_hashes"]["validation_manifest"]:
        raise ValueError("frozen validation manifest hash mismatch")

    source_features = torch.load(source_cache_path, map_location="cpu", weights_only=True)
    all_rows = [("train", train_rows), ("validation", validation_rows)]
    all_ids = [str(row["id"]) for _, rows in all_rows for row in rows]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("duplicate audio sample IDs across splits")
    train_media = {str(row["media_sha256"]) for row in train_rows}
    validation_media = {str(row["media_sha256"]) for row in validation_rows}
    train_speakers = {str(row["speaker_group_sha256"]) for row in train_rows}
    validation_speakers = {str(row["speaker_group_sha256"]) for row in validation_rows}
    if train_media & validation_media or train_speakers & validation_speakers:
        raise ValueError("audio media or speaker identity overlaps frozen splits")
    if source_features.get("train_ids") != [str(row["id"]) for row in train_rows]:
        raise ValueError("source cache train ID order changed")
    if source_features.get("validation_ids") != [str(row["id"]) for row in validation_rows]:
        raise ValueError("source cache validation ID order changed")
    source_features_by_split = {
        "train": source_features["train_features"].float(),
        "validation": source_features["validation_features"].float(),
    }

    encoder_revision = str(config["encoder"]["audio_revision"])
    encoder_weights_sha256 = str(config["encoder"]["audio_weights_sha256"])
    audio_model_path = Path(config["encoder"]["audio_model_path"]).resolve()
    preprocessor_config_path = audio_model_path / "preprocessor_config.json"
    extraction_script_path = Path("scripts/train_frozen_audio_probe.py").resolve()
    if sha256_file(audio_model_path / "model.safetensors") != encoder_weights_sha256:
        raise ValueError("pinned Whisper weight hash mismatch")
    preprocessor_config_sha256 = sha256_file(preprocessor_config_path)
    extraction_code_sha256 = sha256_file(extraction_script_path)
    preprocessing_spec = {
        "sample_rate": 16000,
        "padding": "max_length",
        "max_length": "processor.n_samples",
        "truncation": True,
        "valid_tokens": "min(hidden_tokens,max(1,ceil(waveform_samples/320)))",
        "pooling": "mean over valid last_hidden_state tokens in float32",
        "preprocessor_config_sha256": preprocessor_config_sha256,
        "feature_extraction_code_sha256": extraction_code_sha256,
    }
    preprocessing_sha256 = hashlib.sha256(
        json.dumps(preprocessing_spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    cache_root = output / "feature-cache"
    output.mkdir(parents=True, exist_ok=False)
    observation_cache = ObservationFeatureCache(cache_root)
    reloaded_by_split: dict[str, list[torch.Tensor]] = {"train": [], "validation": []}
    rows_manifest: list[dict[str, Any]] = []
    key_ids: set[str] = set()
    cache_started = time.perf_counter()
    for split_name, rows in all_rows:
        original_features = source_features_by_split[split_name]
        if original_features.shape != (len(rows), 384) or not torch.isfinite(
            original_features
        ).all():
            raise ValueError(f"invalid {split_name} Whisper feature matrix")
        split_loaded = []
        for index, row in enumerate(rows):
            media_path = sample_dir / str(row["media_path"])
            if not media_path.is_file() or sha256_file(media_path) != row["media_sha256"]:
                raise ValueError(f"missing or changed {split_name} audio media: {row['id']}")
            feature = original_features[index].contiguous()
            key = audio_observation_feature_key(
                row,
                encoder_revision=encoder_revision,
                encoder_weights_sha256=encoder_weights_sha256,
                extraction_code_sha256=extraction_code_sha256,
                preprocessor_config_sha256=preprocessor_config_sha256,
                preprocessing_sha256=preprocessing_sha256,
            )
            stored = observation_cache.put(key, feature.numpy().tobytes())
            reloaded = observation_cache.get(key)
            loaded_feature = _tensor_from_payload(reloaded.payload, key.feature_shape)
            if not torch.equal(feature, loaded_feature):
                raise ValueError(f"persisted audio feature differs for {row['id']}")
            if key.cache_id in key_ids:
                raise ValueError("multiple audio sample rows resolved to the same observation key")
            key_ids.add(key.cache_id)
            split_loaded.append(loaded_feature)
            rows_manifest.append(
                {
                    "sample_id": str(row["id"]),
                    "split": split_name,
                    "cache_id": key.cache_id,
                    "speaker_group_sha256": row["speaker_group_sha256"],
                    "media_sha256": row["media_sha256"],
                    "feature_payload_sha256": stored.payload_sha256,
                }
            )
        reloaded_by_split[split_name] = split_loaded
    cache_seconds = time.perf_counter() - cache_started
    reloaded_train = torch.stack(reloaded_by_split["train"])
    reloaded_validation = torch.stack(reloaded_by_split["validation"])
    if not torch.equal(reloaded_train, source_features_by_split["train"]):
        raise ValueError("training feature ordering or content changed after cache reload")
    if not torch.equal(reloaded_validation, source_features_by_split["validation"]):
        raise ValueError("validation feature ordering or content changed after cache reload")

    text_model_path = Path(config["encoder"]["text_model_path"]).resolve()
    query_texts = {
        f"Question: {spec['question']}"
        for label in LABELS
        for spec in build_query_specs(label)
    }
    candidate_texts = {
        f"Candidate answer: {option}."
        for label in LABELS
        for spec in build_query_specs(label)
        for option in spec["options"]
    }
    inference_started = time.perf_counter()
    text_vectors, text_parameter_count = _encode_texts(
        sorted(query_texts | candidate_texts), text_model_path, torch.device("cpu")
    )
    if text_parameter_count != int(config["encoder"]["text_parameters"]):
        raise ValueError("reloaded MiniLM parameter count mismatch")
    validation_examples = make_query_examples(
        validation_rows, reloaded_validation, text_vectors
    )
    input_size = int(validation_examples[0]["features"].shape[-1])
    scorer = CandidateScorer(input_size)
    scorer.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    metrics, predictions = _evaluate(
        scorer, validation_examples, torch.device("cpu"), int(config["readout"]["batch_queries"])
    )
    inference_seconds = time.perf_counter() - inference_started
    expected_predictions = [
        json.loads(line)
        for line in expected_predictions_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    max_logit_delta, max_probability_delta = _compare_predictions(
        expected_predictions, predictions
    )
    metric_keys = ("accuracy", "nll", "brier", "ece_15_bins")
    max_metric_delta = max(
        abs(float(metrics[group][metric]) - float(report["validation_metrics"][group][metric]))
        for group in ("macro_question_type",)
        for metric in metric_keys
    )
    max_metric_delta = max(
        max_metric_delta,
        max(
            abs(
                float(metrics["by_question_type"][task][metric])
                - float(report["validation_metrics"]["by_question_type"][task][metric])
            )
            for task in QUESTION_TASKS
            for metric in metric_keys
        ),
    )
    question_counts = Counter(prediction["sample_id"] for prediction in predictions)
    if set(question_counts.values()) != {len(QUESTION_TASKS)}:
        raise ValueError("audio observation was not reused by exactly four query types")

    manifest_path = output / "cache-records.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows_manifest),
        encoding="utf-8",
    )
    cache_entries = list(cache_root.glob("*.feature"))
    report_output = {
        "schema_version": 1,
        "status": "complete_audio_observation_cache_validation",
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "multiquery_source_run_report_sha256": sha256_file(report_path),
        "source_train_manifest_sha256": sha256_file(train_path),
        "source_validation_manifest_sha256": sha256_file(validation_path),
        "source_feature_cache_sha256": sha256_file(source_cache_path),
        "whisper_weights_sha256": encoder_weights_sha256,
        "whisper_revision": encoder_revision,
        "whisper_parameter_count": int(config["encoder"]["audio_parameters"]),
        "preprocessor_config_sha256": preprocessor_config_sha256,
        "feature_extraction_code_sha256": extraction_code_sha256,
        "preprocessing_spec": preprocessing_spec,
        "preprocessing_sha256": preprocessing_sha256,
        "train_sample_count": len(train_rows),
        "validation_sample_count": len(validation_rows),
        "train_ordered_ids_sha256": _ordered_ids_sha256(train_rows),
        "validation_ordered_ids_sha256": _ordered_ids_sha256(validation_rows),
        "train_unique_audio_assets": len({row["media_sha256"] for row in train_rows}),
        "validation_unique_audio_assets": len({row["media_sha256"] for row in validation_rows}),
        "train_unique_speakers": len({row["speaker_group_sha256"] for row in train_rows}),
        "validation_unique_speakers": len({row["speaker_group_sha256"] for row in validation_rows}),
        "cross_split_audio_asset_overlap": 0,
        "cross_split_speaker_overlap": 0,
        "cache_entries": len(cache_entries),
        "cache_payload_bytes": len(rows_manifest) * 384 * 4,
        "cache_total_bytes": sum(item.stat().st_size for item in cache_entries),
        "cache_manifest_sha256": sha256_file(manifest_path),
        "cache_manifest_path": str(manifest_path),
        "cache_directory": str(cache_root),
        "cache_write_and_reload_seconds": cache_seconds,
        "query_types_per_audio_observation": len(QUESTION_TASKS),
        "validation_queries": len(predictions),
        "validation_queries_by_type": {
            task: sum(prediction["question_type"] == task for prediction in predictions)
            for task in QUESTION_TASKS
        },
        "reloaded_validation_metrics": metrics,
        "saved_run_validation_metrics": report["validation_metrics"],
        "max_abs_logit_delta_vs_saved_run": max_logit_delta,
        "max_abs_probability_delta_vs_saved_run": max_probability_delta,
        "max_metric_delta_vs_saved_run": max_metric_delta,
        "prediction_indices_exact_vs_saved_run": True,
        "cross_backend_numeric_deltas_are_descriptive": True,
        "cpu_validation_inference_seconds": inference_seconds,
        "device": device_name,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
        },
        "sealed_audit_loaded": False,
        "test_split_loaded": False,
        "encoder_executions_in_this_verification": 0,
        "training_performed": False,
        "notes": (
            "Existing train/validation tensors and checkpoint were read only. Every audio asset "
            "was SHA-256 checked before use. One observation cache entry is shared across four "
            "question types; no audio encoder was loaded or executed."
        ),
    }
    report_path_out = output / "verification-report.json"
    report_path_out.write_text(
        json.dumps(report_output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if report_path.read_bytes() != original_report:
        raise AssertionError("source multi-query run report changed during verification")
    print(json.dumps(report_output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
