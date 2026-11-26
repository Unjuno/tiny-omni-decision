"""Score a fixed CLEVRER readout on forward and frame-reversed scene features."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import AutoModel, AutoTokenizer

from scripts.evaluate_clevrer_binary_reasoning_controls import (
    _load_feature_map,
    _paired_scene_bootstrap,
    _prediction_rows,
    _read_jsonl,
)
from scripts.train_frozen_clevrer_binary_reasoning import _decision_metrics, _validate_inputs
from scripts.train_frozen_text_probe import CandidateScorer
from scripts.train_frozen_video_probe import _combine, _predict, _text_features
from tiny_omni_decision.dataset import sha256_file


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--forward-features", type=Path, required=True)
    parser.add_argument("--reverse-features", type=Path, required=True)
    parser.add_argument("--reference-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_cache(
    feature_path: Path,
    sample_dir: Path,
    validation_rows: list[dict[str, Any]],
    *,
    temporal_order: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    report_path = feature_path.with_name("cache-report.json")
    report = _read_json(report_path)
    manifest_hashes = {
        "train": sha256_file(sample_dir / "train.jsonl"),
        "validation": sha256_file(sample_dir / "validation.jsonl"),
        "fetch_report": sha256_file(sample_dir / "fetch-report.json"),
    }
    # The first forward cache predates the explicit temporal_order field; the
    # extraction code at that revision retained decoder order by construction.
    observed_order = report.get("temporal_order", "forward")
    if (
        report.get("status") != "complete_videoprism_frozen_feature_cache"
        or report.get("checkpoint_sha256")
        != "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
        or report.get("dataset_revision")
        != "MIT-IBM/CLEVRER@98b842082ba4f7c18b6b9e3f39145871782a65ef"
        or report.get("sample_manifest_sha256") != manifest_hashes
        or report.get("feature_cache_sha256") != sha256_file(feature_path)
        or observed_order != temporal_order
    ):
        raise ValueError(f"{temporal_order} feature cache provenance mismatch")
    feature_map = _load_feature_map(feature_path)
    validation_media = {str(row["media_sha256"]) for row in validation_rows}
    witness_rows = _read_jsonl(sample_dir / "train.jsonl")
    witness_media = {str(row["media_sha256"]) for row in witness_rows}
    if len(witness_rows) != 1 or validation_media & witness_media:
        raise ValueError("expected one separate extractor witness asset")
    if set(feature_map) != validation_media | witness_media:
        raise ValueError(f"{temporal_order} cache assets do not match the frozen sample")
    return feature_map, report


def main() -> None:
    args = _args()
    wall_started = time.perf_counter()
    run_started = datetime.now(UTC).isoformat()
    config_path = args.config.resolve(strict=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_inputs(config_path, config)
    sample_dir = args.sample_dir.resolve(strict=True)
    rows = _read_jsonl(sample_dir / "validation.jsonl")
    if not rows or any(row.get("split") != "validation" for row in rows):
        raise ValueError("temporal-order control requires only validation rows")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate validation IDs")
    if {row["question_type"] for row in rows} != {
        "explanatory",
        "predictive",
        "counterfactual",
    }:
        raise ValueError("validation must contain all three CLEVRER reasoning types")

    forward_path = args.forward_features.resolve(strict=True)
    reverse_path = args.reverse_features.resolve(strict=True)
    forward_by_hash, forward_report = _validate_cache(
        forward_path, sample_dir, rows, temporal_order="forward"
    )
    reverse_by_hash, reverse_report = _validate_cache(
        reverse_path, sample_dir, rows, temporal_order="reverse"
    )
    forward_video = torch.from_numpy(
        np.stack([forward_by_hash[str(row["media_sha256"])] for row in rows]).copy()
    )
    reverse_video = torch.from_numpy(
        np.stack([reverse_by_hash[str(row["media_sha256"])] for row in rows]).copy()
    )
    del forward_by_hash, reverse_by_hash

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(
        config["text_encoder"]["model_path"], local_files_only=True
    )
    text_encoder = AutoModel.from_pretrained(
        config["text_encoder"]["model_path"], local_files_only=True
    ).eval().to(device)
    for parameter in text_encoder.parameters():
        parameter.requires_grad_(False)
    text_started = time.perf_counter()
    text_features, targets = _text_features(rows, tokenizer, text_encoder, device)
    text_encoding_seconds = time.perf_counter() - text_started
    del tokenizer, text_encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    checkpoint = Path(config["outputs"]["run_dir"]).resolve(strict=True) / "best-readout.pt"
    model = CandidateScorer(2304, hidden_size=128).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    forward_features = _combine(forward_video, text_features)
    reverse_features = _combine(reverse_video, text_features)
    del forward_video, reverse_video, text_features
    inference_started = time.perf_counter()
    forward_logits = _predict(model, forward_features, device)
    reverse_logits = _predict(model, reverse_features, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started

    reference_dir = args.reference_results.resolve(strict=True)
    reference_report_path = reference_dir / "report.json"
    reference_report = _read_json(reference_report_path)
    reference_predictions_path = reference_dir / "validation-predictions.jsonl"
    if sha256_file(reference_predictions_path) != reference_report["artifact_hashes"][
        "validation_predictions"
    ]:
        raise ValueError("saved forward prediction artifact hash differs from its report")
    saved_all_predictions = _read_jsonl(reference_predictions_path)
    saved_forward = [
        row for row in saved_all_predictions if row.get("model") == "videoprism_plus_text_fixed_v0"
    ]
    if [row["id"] for row in saved_forward] != [row["id"] for row in rows]:
        raise ValueError("saved forward predictions do not match validation ID order")

    forward_prediction_rows = _prediction_rows(
        rows, forward_logits, targets, "videoprism_plus_text_forward"
    )
    reverse_prediction_rows = _prediction_rows(
        rows, reverse_logits, targets, "videoprism_plus_text_reversed"
    )
    max_saved_probability_delta = max(
        abs(float(left) - float(right))
        for current, saved in zip(forward_prediction_rows, saved_forward, strict=True)
        for left, right in zip(current["probabilities"], saved["probabilities"], strict=True)
    )
    if max_saved_probability_delta > 1e-6:
        raise ValueError(
            "recomputed forward path does not match the saved baseline probabilities: "
            f"max delta={max_saved_probability_delta}"
        )

    forward_metrics = _decision_metrics(forward_logits, targets, rows)
    reverse_metrics = _decision_metrics(reverse_logits, targets, rows)
    bootstrap_started = time.perf_counter()
    bootstrap = _paired_scene_bootstrap(
        {
            "forward": forward_prediction_rows,
            "frame_reversed": reverse_prediction_rows,
        },
        rows,
        replicates=args.bootstrap_replicates,
        seed=17,
    )
    bootstrap_seconds = time.perf_counter() - bootstrap_started

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    output_dir.mkdir(parents=True)
    predictions_path = output_dir / "validation-predictions.jsonl"
    predictions = [*forward_prediction_rows, *reverse_prediction_rows]
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions),
        encoding="utf-8",
        newline="\n",
    )
    report = {
        "schema_version": 1,
        "status": "complete_clevrer_temporal_order_control",
        "claim_scope": (
            "fixed readout on a development validation set already evaluated once; "
            "temporal-order sensitivity diagnostic, not a new or blind benchmark"
        ),
        "run_started_utc": run_started,
        "run_finished_utc": datetime.now(UTC).isoformat(),
        "sealed_audit_loaded": False,
        "checkpoint_selection_or_training": False,
        "seed": 17,
        "control": {
            "intervention": (
                "reverse the order of the same eight uniformly sampled frames per scene"
            ),
            "frame_count": 8,
            "frame_size": 288,
            "preprocessing_unchanged": True,
            "video_encoder_frozen": True,
            "text_encoder_frozen": True,
            "readout_frozen": True,
            "validation_ids_and_order_unchanged": True,
        },
        "validation": {
            "source_revision": "MIT-IBM/CLEVRER@98b842082ba4f7c18b6b9e3f39145871782a65ef",
            "rows": len(rows),
            "scenes": len({int(row["scene_index"]) for row in rows}),
            "assets": len({str(row["media_sha256"]) for row in rows}),
            "question_type_counts": {
                kind: sum(row["question_type"] == kind for row in rows)
                for kind in ("explanatory", "predictive", "counterfactual")
            },
            "validation_jsonl_sha256": sha256_file(sample_dir / "validation.jsonl"),
            "fetch_report_sha256": sha256_file(sample_dir / "fetch-report.json"),
        },
        "models": {
            "videoprism_revision": "be719a406d563b66f0ac969e7c94bab8e997c81a",
            "videoprism_source_commit": "c6eba0ece3dd49a24ffc44582966997a4a59f1d8",
            "videoprism_checkpoint_sha256": (
                "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
            ),
            "text_revision": config["text_encoder"]["revision"],
            "text_weights_sha256": config["text_encoder"]["weights_sha256"],
            "readout_sha256": sha256_file(checkpoint),
        },
        "forward": {
            "metrics": forward_metrics,
            "feature_cache_sha256": sha256_file(forward_path),
            "feature_cache_report_sha256": sha256_file(
                forward_path.with_name("cache-report.json")
            ),
        },
        "frame_reversed": {
            "metrics": reverse_metrics,
            "feature_cache_sha256": sha256_file(reverse_path),
            "feature_cache_report_sha256": sha256_file(
                reverse_path.with_name("cache-report.json")
            ),
        },
        "paired_scene_bootstrap": bootstrap,
        "baseline_reload_parity": {
            "reference_run_report_sha256": sha256_file(reference_report_path),
            "reference_predictions_sha256": sha256_file(reference_predictions_path),
            "max_forward_probability_delta_vs_saved_control_run": max_saved_probability_delta,
        },
        "timing_seconds": {
            "minilm_feature_encoding": text_encoding_seconds,
            "forward_and_reversed_scorer_evaluation": inference_seconds,
            "paired_scene_bootstrap": bootstrap_seconds,
            "total": time.perf_counter() - wall_started,
        },
        "environment": {
            "torch": torch.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
        },
        "artifacts": {
            "predictions_sha256": sha256_file(predictions_path),
            "evaluator_script_sha256": sha256_file(Path(__file__).resolve()),
            "extractor_script_sha256": sha256_file(
                Path("scripts/extract_frozen_videoprism_features.py").resolve()
            ),
            "source_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
            ).stdout.strip(),
        },
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "forward": forward_metrics,
                      "frame_reversed": reverse_metrics,
                      "paired_scene_bootstrap": bootstrap,
                      "baseline_reload_parity": report["baseline_reload_parity"]}, indent=2))


if __name__ == "__main__":
    main()
