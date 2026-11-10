"""Open the frozen public-validation candidate pool once for final controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.extract_perception_test_text_features import sha256_file
from scripts.train_perception_test_video_readout import (
    TEXT_MODEL_SHA256,
    VIDEO_MODEL_SHA256,
    PerceptionVideoDecisionReadout,
    VideoFeatureCache,
    _metrics,
)

VALID_JSONL_SHA256 = "2090991bf3a956c67e31cb3d3b6cf68e60fbaF29fdf90aae640a110f3c674350".lower()
VALID_ID_ORDER_SHA256 = "6c3c72ac4e0a304f6d6a61726e754155d9153c877d001c9390d96acd9bb191b4"
OPTION_PERMUTATION = [2, 0, 1]
BOOTSTRAP_REPETITIONS = 2_000
SEED = 17


def donor_video_map(video_ids: list[str], *, seed: int = SEED) -> dict[str, str]:
    ids = sorted(set(video_ids))
    if len(ids) != len(video_ids) or len(ids) < 2:
        raise ValueError("unique validation video IDs are required")
    order = np.random.default_rng(seed).permutation(np.asarray(ids, dtype=str)).tolist()
    return {video_id: order[(index + 1) % len(order)] for index, video_id in enumerate(order)}


def _require_fixed_validation_split(rows: list[dict[str, Any]]) -> None:
    if any(row.get("split") != "valid" for row in rows):
        raise ValueError("validation corpus contains an unexpected split")


def _load_validation_rows(corpus_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    path = corpus_dir / "validation.jsonl"
    if (
        sha256_file(path) != VALID_JSONL_SHA256
        or manifest["validation"]["records_sha256"] != VALID_JSONL_SHA256
    ):
        raise ValueError("public-validation corpus differs from frozen candidate")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 2886 or len({row.get("id") for row in rows}) != len(rows):
        raise ValueError("unexpected validation count or duplicate IDs")
    _require_fixed_validation_split(rows)
    order_hash = hashlib.sha256("\n".join(row["id"] for row in rows).encode()).hexdigest()
    if order_hash != VALID_ID_ORDER_SHA256:
        raise ValueError("validation sample order differs from frozen candidate")
    for row in rows:
        target = int(row["target_option_index"])
        if len(row.get("options", [])) != 3 or target not in (0, 1, 2):
            raise ValueError(f"invalid validation MCQA row: {row.get('id')}")
        if row["target"] != row["options"][target]:
            raise ValueError(f"validation target alignment mismatch: {row['id']}")
    return rows, manifest


def _load_validation_text_features(path: Path, report_path: Path, rows):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "COMPLETE"
        or report.get("split") != "validation"
        or report.get("records_sha256") != VALID_JSONL_SHA256
        or report.get("model_files_sha256", {}).get("model.safetensors") != TEXT_MODEL_SHA256
        or report.get("labels_used") is not False
        or report.get("official_test_opened") is not False
        or sha256_file(path) != report.get("feature_sha256")
    ):
        raise ValueError("validation text feature report/hash mismatch")
    with np.load(path, allow_pickle=False) as archive:
        sample_ids = archive["sample_ids"].astype(str)
        video_ids = archive["video_ids"].astype(str)
        features = archive["features"]
    if (
        sample_ids.tolist() != [row["id"] for row in rows]
        or video_ids.tolist() != [row["source_video_id"] for row in rows]
        or features.shape != (len(rows), 3, 384)
        or features.dtype != np.float32
        or not np.isfinite(features).all()
    ):
        raise ValueError("validation text features do not align with fixed sample order")
    return features, report


def _metric_by_type(logits, rows):
    targets = np.asarray([int(row["target_option_index"]) for row in rows], dtype=np.int64)
    kinds = [str(row["reasoning_type"]) for row in rows]
    result = {"overall": _metrics(logits, targets)}
    for kind in sorted(set(kinds)):
        mask = np.asarray([value == kind for value in kinds])
        result[kind] = _metrics(logits[mask], targets[mask])
    return result


def _paired_group_bootstrap(
    logits_by_arm, targets, video_ids, *, repetitions=BOOTSTRAP_REPETITIONS
):
    ids = sorted(set(video_ids))
    row_indices = {video_id: np.flatnonzero(np.asarray(video_ids) == video_id) for video_id in ids}
    rng = np.random.default_rng(SEED)
    arms = sorted(logits_by_arm)
    samples = {
        arm: {metric: [] for metric in ("accuracy", "nll", "brier", "ece_15_bins")} for arm in arms
    }
    for _ in range(repetitions):
        chosen_groups = rng.choice(ids, size=len(ids), replace=True)
        indices = np.concatenate([row_indices[video_id] for video_id in chosen_groups])
        for arm in arms:
            current = _metrics(logits_by_arm[arm][indices], targets[indices])
            for metric in samples[arm]:
                samples[arm][metric].append(current[metric])
    result = {}
    for control in ("text_only", "mismatched_video"):
        result[control] = {
            "comparison": "joint_minus_control",
            "resampling_unit": "video group; all questions for each selected group retained",
            "bootstrap_repetitions": repetitions,
            "bootstrap_seed": SEED,
            "metrics": {},
        }
        for metric in samples["joint"]:
            deltas = np.asarray(samples["joint"][metric]) - np.asarray(samples[control][metric])
            result[control]["metrics"][metric] = {
                "delta": float(deltas.mean()),
                "ci_95_percentile": [
                    float(np.quantile(deltas, 0.025)),
                    float(np.quantile(deltas, 0.975)),
                ],
            }
    return result


def _verify_frozen_checkpoint(training_report_path: Path, output_dir: Path):
    training_dir = training_report_path.parent
    lock_path = training_dir / "public-validation-evaluation-lock.json"
    if lock_path.exists():
        raise FileExistsError("public validation already opened or a prior attempt was interrupted")
    training = json.loads(training_report_path.read_text(encoding="utf-8"))
    if (
        training.get("status") != "TRAIN_COMPLETE"
        or training.get("public_validation_opened") is not False
    ):
        raise ValueError("training report is not a frozen pre-validation candidate")
    config_path = training_dir / "effective-config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    declared = config.pop("config_sha256", None)
    actual = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if declared != actual or training.get("config_sha256") != actual:
        raise ValueError("training effective-config hash mismatch")
    checkpoint = training_dir / "best-checkpoint.pt"
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != training.get("best_checkpoint_sha256"):
        raise ValueError("best checkpoint hash mismatch")
    lock = {
        "status": "STARTED",
        "started_utc": datetime.now(UTC).isoformat(),
        "checkpoint_sha256": checkpoint_hash,
        "training_config_sha256": actual,
        "output_dir": str(output_dir),
        "validation_candidate_sha256": VALID_JSONL_SHA256,
        "official_test_opened": False,
        "sealed_audit_opened": False,
    }
    with lock_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(lock, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return training, checkpoint, lock_path, lock


def evaluate(
    *,
    corpus_dir: Path,
    text_features: Path,
    text_report: Path,
    video_feature_dir: Path,
    video_feature_manifest: Path,
    training_report: Path,
    output_dir: Path,
    device_name: str = "cuda",
) -> dict[str, Any]:
    for path in (
        corpus_dir,
        text_features,
        text_report,
        video_feature_dir,
        video_feature_manifest,
        training_report,
    ):
        path.resolve(strict=True)
    if output_dir.drive.upper() != "D:" or output_dir.exists():
        raise ValueError("use a new, nonexisting evaluation output directory on D:")
    if device_name not in {"cuda", "cpu"} or (
        device_name == "cuda" and not torch.cuda.is_available()
    ):
        raise ValueError("requested evaluation device is unavailable")
    training, checkpoint_path, lock_path, lock = _verify_frozen_checkpoint(
        training_report, output_dir
    )

    started = time.perf_counter()
    rows, corpus_manifest = _load_validation_rows(corpus_dir)
    if corpus_manifest["group_integrity"]["train_validation_video_id_overlap"] != 0:
        raise ValueError("frozen source train/validation groups overlap")
    text, text_meta = _load_validation_text_features(text_features, text_report, rows)
    video_store = VideoFeatureCache(
        video_feature_dir,
        video_feature_manifest,
        expected_split="valid",
    )
    video_ids = [str(row["source_video_id"]) for row in rows]
    if set(video_store.receipts) != set(video_ids):
        raise ValueError("validation VideoMAE feature IDs differ from candidate corpus")
    donors = donor_video_map(sorted(set(video_ids)))
    device = torch.device(device_name)
    model = PerceptionVideoDecisionReadout().to(device).eval()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if (
        payload.get("config_sha256") != training["config_sha256"]
        or payload.get("selected_epoch") != training["selected_epoch"]
    ):
        raise ValueError("checkpoint metadata does not match frozen training report")
    model.load_state_dict(payload["state_dict"], strict=True)
    if (
        sum(parameter.numel() for parameter in model.parameters())
        != training["reader_trainable_parameters"]
    ):
        raise ValueError("reader parameter count differs from training report")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    logits_by_arm = {name: [] for name in ("joint", "text_only", "mismatched_video")}
    permuted_logits = []
    batch_size = 16
    inverse = torch.argsort(torch.tensor(OPTION_PERMUTATION, device=device))
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            batch_rows = rows[start:end]
            videos = np.stack([video_store.get(row["source_video_id"]) for row in batch_rows])
            donor_videos = np.stack(
                [video_store.get(donors[row["source_video_id"]]) for row in batch_rows]
            )
            video_tensor = torch.from_numpy(videos.copy()).to(device=device, dtype=torch.float32)
            donor_tensor = torch.from_numpy(donor_videos.copy()).to(
                device=device, dtype=torch.float32
            )
            query_tensor = torch.from_numpy(text[start:end].copy()).to(
                device=device, dtype=torch.float32
            )
            joint = model(video_tensor, query_tensor)
            text_only = model(torch.zeros_like(video_tensor), query_tensor)
            mismatched = model(donor_tensor, query_tensor)
            permuted = model(video_tensor, query_tensor[:, OPTION_PERMUTATION])[:, inverse]
            logits_by_arm["joint"].append(joint.float().cpu().numpy())
            logits_by_arm["text_only"].append(text_only.float().cpu().numpy())
            logits_by_arm["mismatched_video"].append(mismatched.float().cpu().numpy())
            permuted_logits.append(permuted.float().cpu().numpy())

    logits_by_arm = {name: np.concatenate(values) for name, values in logits_by_arm.items()}
    permuted_logits = np.concatenate(permuted_logits)
    targets = np.asarray([int(row["target_option_index"]) for row in rows], dtype=np.int64)
    metrics = {name: _metric_by_type(values, rows) for name, values in logits_by_arm.items()}
    uniform = {
        "count": len(rows),
        "expected_accuracy": 1.0 / 3.0,
        "nll": math.log(3.0),
        "brier": 2.0 / 3.0,
        "ece_15_bins": 0.0,
    }
    paired = _paired_group_bootstrap(logits_by_arm, targets, video_ids)
    max_permutation_delta = float(np.max(np.abs(logits_by_arm["joint"] - permuted_logits)))
    if max_permutation_delta > 1e-5:
        raise RuntimeError("candidate permutation changed restored option scores")
    donors_hash = hashlib.sha256(
        "\n".join(f"{key}\t{donors[key]}" for key in sorted(donors)).encode("utf-8")
    ).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=False)
    prediction_path = output_dir / "validation-predictions.jsonl"
    with prediction_path.open("x", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(rows):
            arm_predictions = {}
            for arm, values in logits_by_arm.items():
                shifted = values[index] - values[index].max()
                probability = np.exp(shifted)
                probability /= probability.sum()
                arm_predictions[arm] = {
                    "logits": values[index].astype(float).tolist(),
                    "probabilities": probability.astype(float).tolist(),
                    "predicted_option_index": int(probability.argmax()),
                }
            record = {
                "id": row["id"],
                "video_id": row["source_video_id"],
                "reasoning_type": row["reasoning_type"],
                "options": row["options"],
                "target_option_index": int(row["target_option_index"]),
                "target": row["target"],
                "donor_video_id": donors[row["source_video_id"]],
                "arms": arm_predictions,
            }
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())

    report = {
        "schema_version": 1,
        "status": "ONE_SHOT_PUBLIC_VALIDATION_COMPLETE",
        "scope": "development-candidate video-plus-text MCQA; not blind generalization",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip(),
        "evaluation_script_sha256": sha256_file(Path(__file__).resolve()),
        "command_line": sys.argv,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "source_revision": corpus_manifest["source"]["revision"],
        "validation_records_sha256": VALID_JSONL_SHA256,
        "validation_sample_id_order_sha256": VALID_ID_ORDER_SHA256,
        "validation_count": len(rows),
        "validation_video_group_count": len(set(video_ids)),
        "reasoning_type_counts": {
            kind: sum(row["reasoning_type"] == kind for row in rows)
            for kind in sorted({row["reasoning_type"] for row in rows})
        },
        "train_validation_video_overlap": 0,
        "participant_or_script_independence_proven": False,
        "encoders": {
            "videomae_checkpoint_sha256": VIDEO_MODEL_SHA256,
            "minilm_checkpoint_sha256": TEXT_MODEL_SHA256,
            "video_parameters": 22_153_286,
            "text_parameters": 17_389_824,
        },
        "reader": {
            "trainable_parameters": training["reader_trainable_parameters"],
            "selected_epoch": training["selected_epoch"],
            "checkpoint_sha256": training["best_checkpoint_sha256"],
            "training_config_sha256": training["config_sha256"],
        },
        "metrics_by_arm_and_reasoning_type": metrics,
        "uniform_choice_expected_baseline": uniform,
        "paired_video_group_bootstrap": paired,
        "option_permutation": {
            "permutation": OPTION_PERMUTATION,
            "maximum_absolute_logit_difference_after_restoring_order": max_permutation_delta,
            "prediction_flip_count": int(
                (logits_by_arm["joint"].argmax(1) != permuted_logits.argmax(1)).sum()
            ),
        },
        "donor_video_mapping_sha256": donors_hash,
        "text_feature_sha256": text_meta["feature_sha256"],
        "video_feature_manifest_sha256": sha256_file(video_feature_manifest),
        "prediction_file": prediction_path.name,
        "prediction_sha256": sha256_file(prediction_path),
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        "official_test_opened": False,
        "sealed_audit_opened": False,
        "candidate_exposure_not_proven_blind": True,
        "checkpoint_selection_used_only_train_video_groups": True,
    }
    report_path = output_dir / "evaluation-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lock.update(
        {
            "status": "COMPLETE",
            "completed_utc": datetime.now(UTC).isoformat(),
            "evaluation_report_sha256": sha256_file(report_path),
            "prediction_sha256": report["prediction_sha256"],
        }
    )
    temporary_lock = lock_path.with_suffix(".json.tmp")
    temporary_lock.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_lock.replace(lock_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--text-report", type=Path, required=True)
    parser.add_argument("--video-feature-dir", type=Path, required=True)
    parser.add_argument("--video-feature-manifest", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    report = evaluate(
        corpus_dir=args.corpus_dir,
        text_features=args.text_features,
        text_report=args.text_report,
        video_feature_dir=args.video_feature_dir,
        video_feature_manifest=args.video_feature_manifest,
        training_report=args.training_report,
        output_dir=args.output_dir,
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "metrics_by_arm": {
                    arm: values["overall"]
                    for arm, values in report["metrics_by_arm_and_reasoning_type"].items()
                },
                "prediction_sha256": report["prediction_sha256"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
