"""Compare raw-audio Path A features with the frozen TRAIN feature cache on CPU."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from scripts.smoke_path_a_textfree_audio_runtime import (
    SOURCE,
    _option_payload,
    _training_examples,
)
from tiny_omni_decision.textfree_audio_runtime import (
    AUDIO_ENCODER_REVISION,
    AUDIO_ENCODER_SHA256,
    AUDIO_READOUT_SHA256,
    TextFreeAudioRuntime,
    sha256_file,
)

FEATURE_CACHE_SHA256 = "ace323458e7f443daf63382d2d41d99151d2a5801c671d30b0df68d52338ae64"


def _tensor_delta(reference: torch.Tensor, observed: torch.Tensor) -> dict[str, float | bool]:
    if reference.shape != observed.shape:
        raise ValueError("reference and observed tensors must have equal shapes")
    difference = (reference.float() - observed.float()).abs()
    return {
        "max_abs_delta": float(difference.max().item()),
        "mean_abs_delta": float(difference.mean().item()),
        "allclose_1e-5": bool(torch.allclose(reference, observed, rtol=1e-5, atol=1e-5)),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    sample_dir = args.sample_dir.resolve(strict=True)
    feature_cache_path = args.feature_cache.resolve(strict=True)
    encoder_dir = args.encoder_dir.resolve(strict=True)
    readout_path = args.readout.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory must be new: {output_dir}")
    if sha256_file(feature_cache_path) != FEATURE_CACHE_SHA256:
        raise ValueError("frozen feature-cache SHA-256 mismatch")

    rows = _training_examples(sample_dir, args.examples)
    cache = torch.load(feature_cache_path, map_location="cpu", weights_only=True)
    train_manifest = [
        json.loads(line)
        for line in (sample_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    train_ids = [str(row["id"]) for row in train_manifest]
    if cache.get("source_revision") != SOURCE or cache.get("encoder_revision") != (
        AUDIO_ENCODER_REVISION.rsplit("@", 1)[-1]
    ):
        raise ValueError("frozen feature-cache source or encoder revision mismatch")
    if cache.get("train_ids") != train_ids:
        raise ValueError("feature-cache TRAIN IDs/order differ from the frozen manifest")
    cached_features = cache.get("train_features")
    if not isinstance(cached_features, torch.Tensor) or cached_features.shape != (
        len(train_ids), 384
    ):
        raise ValueError("frozen TRAIN feature matrix has an unexpected shape")
    index = {sample_id: row_index for row_index, sample_id in enumerate(train_ids)}

    torch.set_num_threads(4)
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    runtime = TextFreeAudioRuntime(encoder_dir, readout_path, device="cpu")
    records: list[dict[str, object]] = []
    classes = runtime.readout.class_ids
    options, _ = _option_payload(classes)
    for row in rows:
        observed, cache_metrics = runtime.encode_observation(
            sample_dir / row["media_path"], expected_media_sha256=row["media_sha256"]
        )
        reference = cached_features[index[row["id"]]].float()
        reference_decision = runtime.readout.decide(reference, options)
        observed_decision = runtime.readout.decide(observed.to(runtime.device), options)
        reference_probabilities = [item["probability"] for item in reference_decision["options"]]
        observed_probabilities = [item["probability"] for item in observed_decision["options"]]
        probability_delta = _tensor_delta(
            torch.tensor(reference_probabilities), torch.tensor(observed_probabilities)
        )
        records.append(
            {
                "sample_id": row["id"],
                "speaker_group_sha256": row["speaker_group_sha256"],
                "media_sha256": row["media_sha256"],
                "cached_feature_vs_raw_runtime": _tensor_delta(reference, observed),
                "cached_vs_raw_runtime_probabilities": probability_delta,
                "cached_top_class": reference_decision["selected_option_id"],
                "runtime_top_class": observed_decision["selected_option_id"],
                "top_class_same": reference_decision["selected_option_id"]
                == observed_decision["selected_option_id"],
                "encoder_metrics": cache_metrics,
            }
        )
    runtime.close()
    output_dir.mkdir(parents=True)
    strict_feature_parity = all(
        bool(row["cached_feature_vs_raw_runtime"]["allclose_1e-5"]) for row in records
    )
    max_probability_delta = max(
        float(row["cached_vs_raw_runtime_probabilities"]["max_abs_delta"])
        for row in records
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "status": (
            "PASS_CPU_FEATURE_PARITY" if strict_feature_parity else "CPU_BACKEND_DRIFT_OBSERVED"
        ),
        "scope": (
            "CPU raw-WAV inference versus existing frozen TRAIN features; "
            "no labels or quality metrics"
        ),
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "device": "cpu",
        "gpu_used": False,
        "sealed_audit_loaded": False,
        "validation_manifest_opened": False,
        "target_labels_accessed": False,
        "source": SOURCE,
        "train_manifest_sha256": sha256_file(sample_dir / "train.jsonl"),
        "feature_cache_sha256": sha256_file(feature_cache_path),
        "encoder_revision": AUDIO_ENCODER_REVISION,
        "encoder_sha256": AUDIO_ENCODER_SHA256,
        "readout_sha256": AUDIO_READOUT_SHA256,
        "code_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "examples": len(records),
        "unique_speakers": len({str(row["speaker_group_sha256"]) for row in records}),
        "strict_feature_parity": strict_feature_parity,
        "all_top_classes_unchanged": all(bool(row["top_class_same"]) for row in records),
        "max_probability_delta": max_probability_delta,
        "records": records,
    }
    report_path = output_dir / "parity-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report["report_sha256"] = sha256_file(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--encoder-dir", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=4)
    args = parser.parse_args()
    if args.examples < 1 or args.examples > 8:
        parser.error("--examples must be between 1 and 8")
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
