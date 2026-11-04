"""Replay a frozen CLEVRER VideoPrism binary scorer on its saved validation IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tiny_omni_decision.video_runtime import (
    READOUT_SHA256,
    VIDEO_MODEL_PARAMETERS,
    VIDEO_MODEL_SHA256,
    ClevrerBinaryVideoFeatureRuntime,
    sha256_file,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-model-dir", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--feature-report", type=Path, required=True)
    parser.add_argument("--validation-rows", type=Path, required=True)
    parser.add_argument("--expected-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256_text(items: list[str]) -> str:
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()


def main() -> None:
    args = _args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite video runtime output: {output}")
    model_dir = args.text_model_dir.resolve(strict=True)
    readout = args.readout.resolve(strict=True)
    feature_cache = args.feature_cache.resolve(strict=True)
    feature_report_path = args.feature_report.resolve(strict=True)
    rows = _read_jsonl(args.validation_rows.resolve(strict=True))
    expected = _read_jsonl(args.expected_predictions.resolve(strict=True))
    if len(rows) != len(expected) or not rows:
        raise ValueError("validation rows and saved predictions must be nonempty and equal length")
    row_ids = [str(row["id"]) for row in rows]
    expected_ids = [str(row["id"]) for row in expected]
    if len(set(row_ids)) != len(row_ids) or row_ids != expected_ids:
        raise ValueError("validation row IDs must be unique and match saved prediction order")
    if sha256_file(readout) != READOUT_SHA256:
        raise ValueError("readout checkpoint hash mismatch")

    feature_report = json.loads(feature_report_path.read_text(encoding="utf-8"))
    if (
        feature_report.get("status") != "complete_videoprism_frozen_feature_cache"
        or feature_report.get("model_revision") != "be719a406d563b66f0ac969e7c94bab8e997c81a"
        or feature_report.get("upstream_source_commit")
        != "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
        or feature_report.get("checkpoint_sha256") != VIDEO_MODEL_SHA256
    ):
        raise ValueError("VideoPrism feature cache provenance mismatch")
    expected_cache_hash = feature_report.get(
        "feature_file_sha256", feature_report.get("feature_cache_sha256")
    )
    if expected_cache_hash != sha256_file(feature_cache):
        raise ValueError("VideoPrism feature cache hash mismatch")
    with np.load(feature_cache, allow_pickle=False) as cached:
        media_hashes = cached["media_sha256"].astype(str).tolist()
        feature_values = np.asarray(cached["features"], dtype=np.float32)
    if len(media_hashes) != len(set(media_hashes)) or feature_values.shape != (
        len(media_hashes),
        768,
    ):
        raise ValueError("feature cache has duplicate media or invalid feature shape")
    if not np.isfinite(feature_values).all():
        raise ValueError("feature cache contains non-finite features")
    feature_by_hash = dict(zip(media_hashes, feature_values, strict=True))
    row_by_id = {str(row["id"]): row for row in rows}
    if any(row.get("split") != "validation" for row in rows):
        raise ValueError("all input decisions must belong to the validation split")
    missing = sorted({str(row["media_sha256"]) for row in rows} - set(feature_by_hash))
    if missing:
        raise ValueError(f"video feature cache is missing {len(missing)} validation media assets")

    load_started = time.perf_counter()
    runtime = ClevrerBinaryVideoFeatureRuntime(
        model_dir,
        readout,
        device=args.device,
    )
    if args.device == "cuda":
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    feature_batch = torch.from_numpy(
        np.stack([feature_by_hash[str(row_by_id[item_id]["media_sha256"])] for item_id in row_ids])
    )
    questions = [str(row.get("source_question", row["question"])) for row in rows]
    statements = [str(row["candidate_statement"]) for row in rows]
    inference_started = time.perf_counter()
    decisions = runtime.decide_many(feature_batch, questions, statements)
    if args.device == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started

    prediction_rows = []
    logit_deltas = []
    probability_deltas = []
    prediction_mismatches = 0
    for row, expected_row, decision in zip(rows, expected, decisions, strict=True):
        probabilities = decision.probabilities.tolist()
        logits = decision.option_logits.tolist()
        expected_logits = np.asarray(expected_row["logits"], dtype=np.float32)
        expected_probabilities = np.asarray(expected_row["probabilities"], dtype=np.float32)
        logit_deltas.extend(np.abs(np.asarray(logits, dtype=np.float32) - expected_logits).tolist())
        probability_deltas.extend(
            np.abs(np.asarray(probabilities, dtype=np.float32) - expected_probabilities).tolist()
        )
        prediction_mismatches += int(
            decision.predicted_index != int(expected_row["prediction_index"])
        )
        prediction_rows.append(
            {
                "id": row["id"],
                "scene_index": row["scene_index"],
                "media_sha256": row["media_sha256"],
                "question_type": row["question_type"],
                "target": row["target"],
                "prediction_index": decision.predicted_index,
                "probabilities": probabilities,
                "logits": logits,
            }
        )

    output.mkdir(parents=True)
    predictions_path = output / "predictions.jsonl"
    predictions_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in prediction_rows
        ),
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "status": "complete_path_a_clevrer_video_feature_runtime_replay",
        "created_utc": datetime.now(UTC).isoformat(),
        "claim_scope": (
            "runtime parity using cached 8-frame VideoPrism embeddings on the existing "
            "development validation; not raw-video end-to-end or new quality evidence"
        ),
        "device": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
        "input_count": len(rows),
        "unique_video_assets": len({str(row["media_sha256"]) for row in rows}),
        "unique_scenes": len({int(row["scene_index"]) for row in rows}),
        "ordered_id_sha256": _sha256_text(row_ids),
        "feature_cache_sha256": sha256_file(feature_cache),
        "feature_report_sha256": sha256_file(feature_report_path),
        "readout_sha256": sha256_file(readout),
        "text_model_sha256": sha256_file(model_dir / "model.safetensors"),
        "video_model_sha256": VIDEO_MODEL_SHA256,
        "runtime_parameter_count_excluding_shared_video_encoder": runtime.parameter_count,
        "unique_runtime_parameters_including_shared_video_encoder": (
            runtime.parameter_count + VIDEO_MODEL_PARAMETERS
        ),
        "prediction_mismatches": prediction_mismatches,
        "max_logit_abs_delta": float(max(logit_deltas, default=0.0)),
        "max_probability_abs_delta": float(max(probability_deltas, default=0.0)),
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "seconds_per_decision": inference_seconds / len(rows),
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated() if args.device == "cuda" else None
        ),
        "predictions_sha256": sha256_file(predictions_path),
        "validation_quality_metrics_recomputed": False,
        "sealed_audit_loaded": False,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
