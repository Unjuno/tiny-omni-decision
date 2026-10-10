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

from scripts.fetch_clevrer_probe import REVISION, TAXONOMIES, VALIDATION_QUESTIONS_SHA256, classify
from scripts.train_frozen_text_probe import _metrics
from scripts.train_frozen_video_probe import (
    FRAME_COUNT,
    _combine,
    _load_rows,
    _predict,
    _text_features,
    _validate_splits,
)
from tiny_omni_decision.dataset import sha256_file
from tiny_omni_decision.decision import FrozenFeatureCandidateScorer

TEXT_REVISION = "4ca70771034acceecb2e72475f72050fcdde4ddc"
TEXT_MODEL_WEIGHTS_SHA256 = "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"
VJEPA_CHECKPOINT_SHA256 = "848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d"


def expand_descriptive_questions(
    raw_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Expand sampled validation scenes while retaining their pinned media identity."""
    media_by_scene: dict[int, dict[str, Any]] = {}
    for row in validation_rows:
        scene = int(row["scene_index"])
        media = {
            key: row[key]
            for key in (
                "media_path",
                "media_sha256",
                "media_bytes",
                "video_filename",
                "scene_group_id",
                "source",
                "split",
            )
            if key in row
        }
        if scene in media_by_scene and media_by_scene[scene] != media:
            raise ValueError(f"inconsistent sampled media identity for scene {scene}")
        media_by_scene[scene] = media

    raw_by_scene = {int(row["scene_index"]): row for row in raw_rows}
    if len(raw_by_scene) != len(raw_rows):
        raise ValueError("raw CLEVRER validation data has duplicate scene rows")
    missing = set(media_by_scene) - set(raw_by_scene)
    if missing:
        raise ValueError(f"raw validation questions missing sampled scenes: {sorted(missing)}")

    expanded: list[dict[str, Any]] = []
    for scene in sorted(media_by_scene):
        raw_scene = raw_by_scene[scene]
        media = media_by_scene[scene]
        if str(raw_scene["video_filename"]) != str(media["video_filename"]):
            raise ValueError(f"raw video filename does not match sampled scene {scene}")
        for question in raw_scene.get("questions", []):
            if question.get("question_type") != "descriptive":
                continue
            taxonomy = question.get("question_subtype")
            target = question.get("answer")
            if taxonomy not in TAXONOMIES or target not in TAXONOMIES[taxonomy]:
                continue
            question_id = int(question["question_id"])
            expanded.append(
                {
                    **media,
                    "id": f"clevrer-validation-{scene:05d}-q{question_id:03d}",
                    "scene_index": scene,
                    "question_id": question_id,
                    "question_type": classify(question),
                    "taxonomy": taxonomy,
                    "question": str(question["question"]),
                    "options": list(TAXONOMIES[taxonomy]),
                    "target": str(target),
                    "program": question.get("program", []),
                }
            )
    expanded.sort(key=lambda row: (row["scene_index"], row["question_id"]))
    ids = [row["id"] for row in expanded]
    if len(ids) != len(set(ids)):
        raise ValueError("expanded CLEVRER question IDs are not unique")
    return expanded


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
    video_rows = torch.stack([cached_video[str(row["media_sha256"])] for row in rows])
    cache_read_seconds = time.perf_counter() - cache_read_started

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
