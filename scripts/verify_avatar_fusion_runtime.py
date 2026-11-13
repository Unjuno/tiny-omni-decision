"""Run a frozen-feature, train-only integration check for the AVATAR scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from tiny_omni_decision.avatar_fusion_runtime import AvatarAVDecisionRuntime


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_verification(
    *,
    checkpoint_path: Path,
    normalization_path: Path,
    features_path: Path,
    manifest_path: Path,
    output_dir: Path,
    expected_checkpoint_sha256: str,
    expected_normalization_sha256: str,
    expected_features_sha256: str,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    expected_files = (
        (checkpoint_path, expected_checkpoint_sha256, "checkpoint"),
        (normalization_path, expected_normalization_sha256, "normalization"),
        (features_path, expected_features_sha256, "features"),
        (manifest_path, expected_manifest_sha256, "manifest"),
    )
    actual_hashes: dict[str, str] = {}
    for path, expected, name in expected_files:
        actual = sha256_file(path)
        if actual != expected.lower():
            raise ValueError(f"{name} SHA-256 mismatch: {actual}")
        actual_hashes[name] = actual

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite runtime verification: {output_dir}")
    output_dir.mkdir(parents=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("manifest must be a nonempty list of fixed examples")
    with np.load(features_path, allow_pickle=False) as archive:
        if set(archive.files) != {"audio", "video"}:
            raise ValueError("feature archive must contain only audio and video arrays")
        audio_features = archive["audio"].copy()
        video_features = archive["video"].copy()
    if audio_features.shape != (len(manifest), 384):
        raise ValueError("audio features do not match manifest rows and width 384")
    if video_features.shape != (len(manifest), 4, 768):
        raise ValueError("video features do not match four options and width 768")
    if not np.isfinite(audio_features).all() or not np.isfinite(video_features).all():
        raise ValueError("feature archive contains non-finite values")

    started = time.perf_counter()
    runtime = AvatarAVDecisionRuntime.load(
        checkpoint_path,
        normalization_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_normalization_sha256=expected_normalization_sha256,
        device="cpu",
    )
    load_seconds = time.perf_counter() - started
    predictions: list[dict[str, Any]] = []
    correct = 0
    max_permutation_delta = 0.0
    inference_calls = 0
    for index, item in enumerate(manifest):
        option_ids = item["option_ids"]
        option_texts = item["option_texts"]
        if len(option_ids) != 4 or len(option_texts) != 4:
            raise ValueError("each AVATAR example must provide four options")
        result = runtime.decide(
            audio_features[index],
            video_features[index],
            option_ids,
            option_texts,
            request_id=item["example_id"],
        )
        inference_calls += 1
        if result["selected_option_id"] == item["target_option_id"]:
            correct += 1

        permutation = [2, 0, 3, 1]
        reordered = runtime.decide(
            audio_features[index],
            video_features[index][permutation],
            [option_ids[position] for position in permutation],
            [option_texts[position] for position in permutation],
            request_id=f"{item['example_id']}-permuted",
        )
        inference_calls += 1
        original_by_id = {option["id"]: option["probability"] for option in result["options"]}
        reordered_by_id = {option["id"]: option["probability"] for option in reordered["options"]}
        delta = max(
            abs(original_by_id[option_id] - reordered_by_id[option_id]) for option_id in option_ids
        )
        max_permutation_delta = max(max_permutation_delta, delta)
        if delta > 1e-6:
            raise ValueError(f"option ordering changed probabilities: {delta}")
        predictions.append(
            {
                "example_id": item["example_id"],
                "target_option_id": item["target_option_id"],
                "prediction": result["selected_option_id"],
                "options": result["options"],
            }
        )

    first = manifest[0]
    missing_audio = runtime.decide(
        None,
        video_features[0],
        first["option_ids"],
        first["option_texts"],
        request_id="missing-audio",
    )
    missing_video = runtime.decide(
        audio_features[0],
        None,
        first["option_ids"],
        first["option_texts"],
        request_id="missing-video",
    )
    if missing_audio.get("status") != "abstain" or missing_video.get("status") != "abstain":
        raise ValueError("runtime must abstain when either required modality is missing")

    prediction_path = output_dir / "predictions.jsonl"
    prediction_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
        encoding="utf-8",
    )
    report = {
        "status": "PASS",
        "classification": "TRAIN_ONLY_RUNTIME_INTEGRATION_NOT_QUALITY_EVALUATION",
        "input_split": "train",
        "sealed_audit_loaded": False,
        "validation_loaded": False,
        "input_hashes": actual_hashes,
        "sample_count": len(manifest),
        "inference_calls_including_permutations": inference_calls,
        "encoder_calls": 0,
        "runtime_parameter_count": runtime.parameter_count,
        "runtime_parameters_frozen": all(
            not parameter.requires_grad for parameter in runtime.parameters()
        ),
        "option_reorder_max_probability_delta": max_permutation_delta,
        "missing_audio_abstained": True,
        "missing_video_abstained": True,
        "train_correct": correct,
        "train_accuracy": correct / len(manifest),
        "train_accuracy_is_not_validation": True,
        "load_seconds": load_seconds,
        "total_wall_seconds": time.perf_counter() - started,
        "raw_media_reextraction": "not performed; inputs are frozen encoder feature vectors",
        "prediction_file_sha256": sha256_file(prediction_path),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-normalization-sha256", required=True)
    parser.add_argument("--expected-features-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()
    result = run_verification(
        checkpoint_path=args.checkpoint,
        normalization_path=args.normalization,
        features_path=args.features,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_normalization_sha256=args.expected_normalization_sha256,
        expected_features_sha256=args.expected_features_sha256,
        expected_manifest_sha256=args.expected_manifest_sha256,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
