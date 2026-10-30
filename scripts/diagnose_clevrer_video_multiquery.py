"""Evaluate more CLEVRER questions over each existing cached validation video."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from scripts.fetch_clevrer_probe import REVISION, VALIDATION_QUESTIONS_SHA256
from scripts.train_frozen_text_probe import _metrics
from scripts.train_frozen_video_probe import (
    FRAME_COUNT,
    _combine,
    _load_rows,
    _predict,
    _text_features,
    _validate_splits,
)
from tiny_omni_decision.cache import ObservationFeatureCache, ObservationFeatureKey
from tiny_omni_decision.clevrer import expand_descriptive_questions
from tiny_omni_decision.dataset import sha256_file
from tiny_omni_decision.decision import FrozenFeatureCandidateScorer

TEXT_REVISION = "4ca70771034acceecb2e72475f72050fcdde4ddc"
TEXT_MODEL_WEIGHTS_SHA256 = "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"
VJEPA_CHECKPOINT_SHA256 = "848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d"
VJEPA_SOURCE_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--raw-validation-questions", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--readout-dir", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument("--text-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_inputs(
    args: argparse.Namespace,
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    raw_validation: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if sha256_file(args.raw_validation_questions) != VALIDATION_QUESTIONS_SHA256:
        raise ValueError("raw CLEVRER validation JSON hash differs from pinned revision")
    if args.text_revision != TEXT_REVISION:
        raise ValueError(f"unexpected MiniLM revision: {args.text_revision}")
    text_weights = args.text_model / "model.safetensors"
    if not text_weights.is_file() or sha256_file(text_weights) != TEXT_MODEL_WEIGHTS_SHA256:
        raise ValueError("local MiniLM weights do not match the pinned SHA-256")
    _validate_splits(args.sample_dir, train, validation)

    readout_report = _read_json(args.readout_dir / "report.json")
    cache_report = _read_json(args.cache_dir / "report.json")
    if readout_report.get("status") != "complete_clevrer_frozen_video_probe":
        raise ValueError("readout artifact is not a completed CLEVRER video probe")
    if cache_report.get("status") != "complete_clevrer_frozen_video_probe":
        raise ValueError("feature cache report is not a completed CLEVRER video probe")

    sample_hashes = {
        "train": sha256_file(args.sample_dir / "train.jsonl"),
        "validation": sha256_file(args.sample_dir / "validation.jsonl"),
        "fetch_report": sha256_file(args.sample_dir / "fetch-report.json"),
    }
    if cache_report.get("sample_manifest_sha256") != sample_hashes:
        raise ValueError("saved sample manifests do not match the feature-cache report")
    if readout_report.get("sample_manifest_sha256") is not None and (
        readout_report.get("sample_manifest_sha256") != sample_hashes
    ):
        raise ValueError("saved sample manifests do not match the readout report")
    for report in (readout_report, cache_report):
        if report.get("frame_count") != FRAME_COUNT:
            raise ValueError("probe frame count differs from the frozen video configuration")
        if report.get("vjepa_checkpoint_sha256") != VJEPA_CHECKPOINT_SHA256:
            raise ValueError("probe report does not match the pinned V-JEPA checkpoint")
        if report.get("text_model_revision") != TEXT_REVISION:
            raise ValueError("probe report does not match the pinned MiniLM revision")

    cache_path = args.cache_dir / "video-features.pt"
    readout_path = args.readout_dir / "best-readout.pt"
    baseline_predictions_path = args.readout_dir / "validation-predictions.jsonl"
    if sha256_file(cache_path) != cache_report["artifact_hashes"]["video-features.pt"]:
        raise ValueError("feature cache file hash differs from its saved report")
    if sha256_file(readout_path) != cache_report["readout_sha256"]:
        raise ValueError("readout file hash differs from the feature-cache run report")
    if (
        sha256_file(baseline_predictions_path)
        != cache_report["artifact_hashes"]["validation-predictions.jsonl"]
    ):
        raise ValueError("baseline predictions differ from the feature-cache run")
    for key in (
        "best_epoch_by_validation_nll",
        "validation_metrics",
        "validation_by_question_type",
    ):
        if readout_report.get(key) != cache_report.get(key):
            raise ValueError(f"readout and cache reports disagree on {key}")

    feature_payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    expected_ids = [row["id"] for row in validation]
    expected_scenes = [int(row["scene_index"]) for row in validation]
    expected_hashes = [str(row["media_sha256"]) for row in validation]
    if feature_payload.get("source_revision") != f"MIT-IBM/CLEVRER@{REVISION}":
        raise ValueError("feature cache source revision mismatch")
    if feature_payload.get("vjepa_checkpoint_sha256") != VJEPA_CHECKPOINT_SHA256:
        raise ValueError("feature cache checkpoint hash mismatch")
    if feature_payload.get("frame_count") != FRAME_COUNT:
        raise ValueError("feature cache frame count mismatch")
    if feature_payload.get("validation_ids") != expected_ids:
        raise ValueError("cached validation IDs or their order differ from the frozen sample")
    if feature_payload.get("validation_scene_ids") != expected_scenes:
        raise ValueError("cached validation scene IDs differ from the frozen sample")
    if feature_payload.get("validation_media_hashes") != expected_hashes:
        raise ValueError("cached validation media hashes differ from the frozen sample")
    if set(feature_payload["validation_video_features_by_media_sha256"]) != set(expected_hashes):
        raise ValueError("cached validation features do not cover exactly the sample media")

    expanded = expand_descriptive_questions(raw_validation, validation)
    baseline = _load_rows(args.readout_dir / "validation-predictions.jsonl")
    baseline_by_id = {row["id"]: row for row in baseline}
    expanded_by_id = {row["id"]: row for row in expanded}
    if len(baseline_by_id) != len(baseline):
        raise ValueError("saved baseline validation predictions have duplicate IDs")
    missing_baseline = set(baseline_by_id) - set(expanded_by_id)
    if missing_baseline:
        raise ValueError("expanded questions do not include every original validation example")
    for row_id, prior in baseline_by_id.items():
        current = expanded_by_id[row_id]
        for key in ("options", "target", "scene_index", "question_type", "taxonomy"):
            if current[key] != prior[key]:
                raise ValueError(f"expanded validation row changed existing {key} for {row_id}")

    if {int(row["scene_index"]) for row in train} & {
        int(row["scene_index"]) for row in expanded
    }:
        raise ValueError("expanded validation scene overlaps the frozen train scenes")
    return feature_payload, readout_report, cache_report


def _group_metrics(
    rows: list[dict[str, Any]], logits: list[torch.Tensor], targets: list[int], key: str
) -> dict[str, dict[str, Any]]:
    groups = sorted({str(row[key]) for row in rows})
    output = {}
    for group in groups:
        indices = [index for index, row in enumerate(rows) if str(row[key]) == group]
        output[group] = {
            "count": len(indices),
            "unique_scenes": len({int(rows[index]["scene_index"]) for index in indices}),
            **_metrics([logits[index] for index in indices], [targets[index] for index in indices]),
        }
    return output


def main() -> None:
    args = _args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    args.sample_dir = args.sample_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.readout_dir = args.readout_dir.resolve()
    args.raw_validation_questions = args.raw_validation_questions.resolve()
    args.text_model = args.text_model.resolve()
    train = _load_rows(args.sample_dir / "train.jsonl")
    validation = _load_rows(args.sample_dir / "validation.jsonl")
    raw_bytes = args.raw_validation_questions.read_bytes()
    raw_validation = json.loads(raw_bytes)
    feature_payload, readout_report, cache_report = _verify_inputs(
        args, train, validation, raw_validation
    )
    rows = expand_descriptive_questions(raw_validation, validation)

    if not torch.cuda.is_available():
        raise RuntimeError("the pinned MiniLM inference path requires the local CUDA device")
    device = torch.device("cuda")
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats(device)
    started = datetime.now(UTC).isoformat()

    cache = feature_payload["validation_video_features_by_media_sha256"]
    cache_read_started = time.perf_counter()
    media_hashes = sorted({str(row["media_sha256"]) for row in rows})
    cached_video = {key: cache[key].float().cpu() for key in media_hashes}
    cache_read_seconds = time.perf_counter() - cache_read_started
    output.mkdir(parents=True, exist_ok=True)
    typed_cache = ObservationFeatureCache(output / "observation-feature-cache")
    media_to_cache_id: dict[str, str] = {}
    typed_cache_entries = []
    validation_scene_by_hash = {str(row["media_sha256"]): row for row in validation}
    av_version = cache_report["environment"]["av"]
    preprocessing_spec = {
        "decoder": "PyAV",
        "decoder_version": av_version,
        "frame_count": FRAME_COUNT,
        "frame_selection": "uniform-linspace-rounded-endpoints-v1",
        "preprocessor": "vjepa2_preprocessor(pretrained=False,crop_size=384)",
        "vjepa_source_commit": VJEPA_SOURCE_COMMIT,
        "feature_extraction_code_sha256": cache_report["training_script_sha256"],
    }
    preprocessing_sha256 = hashlib.sha256(
        json.dumps(preprocessing_spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    video_features_from_persistent_cache: dict[str, torch.Tensor] = {}
    for media_hash in media_hashes:
        row = validation_scene_by_hash[media_hash]
        feature = cached_video[media_hash].contiguous()
        if feature.shape != (768,) or not torch.isfinite(feature).all():
            raise ValueError(f"invalid cached V-JEPA observation feature for {media_hash}")
        key = ObservationFeatureKey(
            modality="video",
            source_id="MIT-IBM/CLEVRER",
            source_revision=REVISION,
            observation_sha256=media_hash,
            encoder_id="facebookresearch/vjepa2/vjepa2_1_vit_base_384",
            encoder_revision=VJEPA_SOURCE_COMMIT,
            encoder_weights_sha256=VJEPA_CHECKPOINT_SHA256,
            preprocessor_id="vjepa2_preprocessor",
            preprocessor_revision=f"{VJEPA_SOURCE_COMMIT}+PyAV-{av_version}",
            preprocessing_sha256=preprocessing_sha256,
            feature_name="mean_spatiotemporal_tokens",
            feature_dtype="float32",
            feature_shape=(768,),
            temporal_policy=f"uniform-linspace-{FRAME_COUNT}-inclusive-v1",
        )
        typed_cache.put(key, feature.numpy().tobytes())
        restored = typed_cache.get(key)
        roundtrip = torch.frombuffer(bytearray(restored.payload), dtype=torch.float32).reshape(768)
        if not torch.equal(feature, roundtrip):
            raise ValueError(f"persistent cache roundtrip changed feature for {media_hash}")
        video_features_from_persistent_cache[media_hash] = roundtrip.clone()
        media_to_cache_id[media_hash] = key.cache_id
        entry_path = output / "observation-feature-cache" / f"{key.cache_id}.feature"
        typed_cache_entries.append(
            {
                "scene_index": int(row["scene_index"]),
                "observation_sha256": media_hash,
                "cache_id": key.cache_id,
                "feature_payload_sha256": restored.payload_sha256,
                "feature_payload_bytes": len(restored.payload),
                "entry_bytes": restored.entry_bytes,
                "entry_sha256": sha256_file(entry_path),
                "key": key.to_dict(),
            }
        )
    video_rows = torch.stack(
        [video_features_from_persistent_cache[str(row["media_sha256"])] for row in rows]
    )
    cache_manifest_path = output / "observation-feature-cache-manifest.json"
    cache_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cache_key_scope": "observation-level; no question or candidate text in key",
                "entries": typed_cache_entries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    text_load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.text_model, local_files_only=True)
    text_encoder = (
        AutoModel.from_pretrained(args.text_model, local_files_only=True).eval().to(device)
    )
    text_load_seconds = time.perf_counter() - text_load_started
    for parameter in text_encoder.parameters():
        parameter.requires_grad_(False)
    text_started = time.perf_counter()
    text_features, targets = _text_features(rows, tokenizer, text_encoder, device)
    torch.cuda.synchronize(device)
    text_seconds = time.perf_counter() - text_started
    del text_encoder, tokenizer
    torch.cuda.empty_cache()

    candidate_features = _combine(video_rows, text_features)
    readout_load_started = time.perf_counter()
    state = torch.load(args.readout_dir / "best-readout.pt", map_location="cpu", weights_only=True)
    first_weight = state["network.0.weight"]
    scorer = FrozenFeatureCandidateScorer(
        feature_size=int(first_weight.shape[1]), hidden_size=int(first_weight.shape[0])
    )
    scorer.load_state_dict(state, strict=True)
    scorer.eval().to(device)
    readout_load_seconds = time.perf_counter() - readout_load_started
    inference_started = time.perf_counter()
    logits = _predict(scorer, candidate_features, device)
    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started

    predictions = []
    for row, item_logits, target_index in zip(rows, logits, targets, strict=True):
        probabilities = torch.softmax(item_logits.float(), dim=-1)
        predictions.append(
            {
                "id": row["id"],
                "scene_index": int(row["scene_index"]),
                "scene_group_id": row["scene_group_id"],
                "media_sha256": row["media_sha256"],
                "observation_feature_cache_id": media_to_cache_id[str(row["media_sha256"])],
                "question_id": int(row["question_id"]),
                "question_type": row["question_type"],
                "taxonomy": row["taxonomy"],
                "question": row["question"],
                "options": row["options"],
                "target": row["options"][target_index],
                "prediction": row["options"][int(probabilities.argmax())],
                "logits": item_logits.tolist(),
                "probabilities": probabilities.tolist(),
            }
        )

    baseline = _load_rows(args.readout_dir / "validation-predictions.jsonl")
    by_prediction_id = {row["id"]: row for row in predictions}
    baseline_max_probability_delta = 0.0
    baseline_prediction_mismatches = 0
    for prior in baseline:
        current = by_prediction_id[prior["id"]]
        delta = max(
            abs(float(left) - float(right))
            for left, right in zip(prior["probabilities"], current["probabilities"], strict=True)
        )
        baseline_max_probability_delta = max(baseline_max_probability_delta, delta)
        baseline_prediction_mismatches += int(prior["prediction"] != current["prediction"])
    if baseline_prediction_mismatches or baseline_max_probability_delta > 1e-5:
        raise ValueError(
            "reloaded readout failed parity on the original 16 validation questions: "
            f"max probability delta={baseline_max_probability_delta}"
        )

    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "validation-predictions.jsonl"
    prediction_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions), encoding="utf-8"
    )
    metric_logits = [torch.tensor(row["logits"], dtype=torch.float32) for row in predictions]
    overall = _metrics(metric_logits, targets)
    scene_counts = Counter(str(row["scene_index"]) for row in rows)
    report = {
        "schema_version": 1,
        "status": "complete_clevrer_validation_multiquery_cache_diagnostic",
        "claim_scope": (
            "read-only diagnostic on the same eight already-used development-validation scenes; "
            "not an independent or sealed evaluation and not a product-quality estimate"
        ),
        "started_utc": started,
        "finished_utc": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "sample_dir": str(args.sample_dir),
        "raw_validation_questions": {
            "path": str(args.raw_validation_questions),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "source_revision": f"MIT-IBM/CLEVRER@{REVISION}",
        },
        "sample_manifest_sha256": {
            "train": sha256_file(args.sample_dir / "train.jsonl"),
            "validation": sha256_file(args.sample_dir / "validation.jsonl"),
            "fetch_report": sha256_file(args.sample_dir / "fetch-report.json"),
        },
        "readout_sha256": sha256_file(args.readout_dir / "best-readout.pt"),
        "readout_legacy_report_sha256": sha256_file(args.readout_dir / "report.json"),
        "readout_artifact_record_sha256": sha256_file(args.cache_dir / "report.json"),
        "readout_artifact_recording_source_commit": cache_report.get("source_commit"),
        "readout_legacy_report_source_commit": readout_report.get("source_commit"),
        "text_encoder_weights_sha256": TEXT_MODEL_WEIGHTS_SHA256,
        "readout_best_epoch_by_validation_nll": readout_report["best_epoch_by_validation_nll"],
        "feature_cache_sha256": sha256_file(args.cache_dir / "video-features.pt"),
        "persistent_observation_cache": {
            "entry_count": len(typed_cache_entries),
            "unique_feature_payload_bytes": sum(
                entry["feature_payload_bytes"] for entry in typed_cache_entries
            ),
            "total_entry_bytes": sum(entry["entry_bytes"] for entry in typed_cache_entries),
            "manifest_path": str(cache_manifest_path),
            "manifest_sha256": sha256_file(cache_manifest_path),
            "all_features_roundtrip_exact": True,
        },
        "feature_cache_source_commit": cache_report.get("source_commit"),
        "source_cache_run_validation_video_encoder_executions": cache_report[
            "video_encoder_executions"
        ]["validation"],
        "source_cache_run_validation_video_encoding_seconds": cache_report["timing_seconds"][
            "validation_video_encoding"
        ],
        "source_cache_run_peak_cuda_allocated_bytes": cache_report[
            "peak_cuda_allocated_bytes"
        ],
        "feature_cache_frame_count": FRAME_COUNT,
        "vjepa_checkpoint_sha256": VJEPA_CHECKPOINT_SHA256,
        "text_model_revision": TEXT_REVISION,
        "text_model_path": str(args.text_model),
        "sealed_audit_loaded": False,
        "training_performed": False,
        "metrics_are_checkpoint_selection_independent": False,
        "selection_caveat": (
            "The same eight validation scenes were used in the original readout's checkpoint "
            "selection; this expansion is descriptive follow-up on those development scenes."
        ),
        "question_count": len(rows),
        "unique_scenes": len(scene_counts),
        "questions_per_scene": dict(sorted(scene_counts.items())),
        "unique_video_assets": len(media_hashes),
        "question_type_counts": dict(sorted(Counter(row["question_type"] for row in rows).items())),
        "taxonomy_counts": dict(sorted(Counter(row["taxonomy"] for row in rows).items())),
        "multiquery_cache": {
            "queries_per_video_mean": len(rows) / len(media_hashes),
            "stored_unique_video_features": len(cache),
            "feature_encoder_executions_in_this_run": 0,
            "cached_encoder_executions_in_original_feature_run": len(media_hashes),
            "hypothetical_per_question_encoder_executions_without_reuse": len(rows),
            "hypothetical_encoder_executions_avoided": len(rows) - len(media_hashes),
            "unique_feature_payload_bytes": sum(
                cached_video[key].numel() * cached_video[key].element_size() for key in media_hashes
            ),
            "expanded_question_feature_payload_bytes": video_rows.numel()
            * video_rows.element_size(),
            "uncached_per_question_runtime_measured": False,
        },
        "baseline_reload_parity_on_original_validation_subset": {
            "count": len(baseline),
            "prediction_mismatches": baseline_prediction_mismatches,
            "max_probability_absolute_delta": baseline_max_probability_delta,
            "tolerance": 1e-5,
        },
        "validation_metrics": overall,
        "validation_by_question_type": _group_metrics(
            rows, metric_logits, targets, "question_type"
        ),
        "validation_by_taxonomy": _group_metrics(rows, metric_logits, targets, "taxonomy"),
        "timing_seconds": {
            "loading_unique_video_features_from_existing_cache": cache_read_seconds,
            "loading_frozen_text_encoder_and_tokenizer": text_load_seconds,
            "frozen_text_encoder": text_seconds,
            "loading_readout": readout_load_seconds,
            "readout_inference": inference_seconds,
        },
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "device": torch.cuda.get_device_name(device),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "transformers": __import__("transformers").__version__,
        },
        "predictions": {
            "path": str(prediction_path),
            "bytes": prediction_path.stat().st_size,
            "sha256": sha256_file(prediction_path),
        },
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "question_count", "unique_scenes", "multiquery_cache",
        "baseline_reload_parity_on_original_validation_subset", "validation_metrics",
        "validation_by_question_type", "validation_by_taxonomy", "timing_seconds",
        "cuda_peak_allocated_bytes",
    )}, indent=2))


if __name__ == "__main__":
    main()
