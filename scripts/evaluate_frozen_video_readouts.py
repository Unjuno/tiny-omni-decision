"""Evaluate selected frozen CLEVRER video readouts without fitting or selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from scripts.audit_videoprism_checkpoint import (
    MODEL_REVISION as VIDEOPRISM_REVISION,
)
from scripts.audit_videoprism_checkpoint import (
    SOURCE_COMMIT as VIDEOPRISM_SOURCE_COMMIT,
)
from scripts.train_frozen_text_probe import CandidateScorer, _metrics
from scripts.train_frozen_video_probe import (
    TEXT_REVISION,
    _combine,
    _load_rows,
    _predict,
    _text_features,
    _validate_splits,
)
from tiny_omni_decision.dataset import sha256_file

VJEPA_CHECKPOINT_SHA256 = "848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d"
VJEPA_READOUT_SHA256 = "0db407b986c8a19e836883f3137d4ed99b684e962c531ba50907e3a6ebfcd25e"
VIDEOPRISM_READOUT_SHA256 = "f68b04a4079c537772938d27c6456841c5eed19595ab49d0f42dd014211f0dbc"
VJEPA_READOUT_ID = "vjepa-clevrer-expanded-selected"
VIDEOPRISM_READOUT_ID = "videoprism-clevrer-expanded-selected"
VIDEOPRISM_CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
VJEPA_SOURCE_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--excluded-scenes-json", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument("--vjepa-feature-cache", type=Path, required=True)
    parser.add_argument("--vjepa-cache-report", type=Path, required=True)
    parser.add_argument("--vjepa-readout", type=Path, required=True)
    parser.add_argument("--videoprism-feature-cache", type=Path, required=True)
    parser.add_argument("--videoprism-cache-report", type=Path, required=True)
    parser.add_argument("--videoprism-readout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def _feature_map(cache_path: Path) -> dict[str, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as payload:
        hashes = payload["media_sha256"].astype(str).tolist()
        values = payload["features"]
    if values.shape != (len(hashes), 768) or not np.isfinite(values).all():
        raise ValueError(f"invalid pooled video feature cache: {cache_path}")
    if len(hashes) != len(set(hashes)):
        raise ValueError(f"duplicate media identities in feature cache: {cache_path}")
    return dict(zip(hashes, values, strict=True))


def _validate_unseen_scenes(validation: list[dict[str, Any]], exclusions: dict[str, Any]) -> None:
    if set(exclusions) != {"train", "validation"}:
        raise ValueError("excluded scene file must contain exactly train and validation lists")
    excluded = exclusions["validation"]
    if not isinstance(excluded, list) or any(
        not isinstance(scene, int) or isinstance(scene, bool) or scene < 0 for scene in excluded
    ):
        raise ValueError("excluded validation scenes must be non-negative integer IDs")
    if len(excluded) != len(set(excluded)):
        raise ValueError("excluded validation scene IDs must be unique")
    heldout_scenes = {int(row["scene_index"]) for row in validation}
    if heldout_scenes & set(excluded):
        raise ValueError("held-out validation includes a previously used video scene")


def _validate_cache_report(
    report_path: Path,
    cache_path: Path,
    sample_dir: Path,
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    expected_status: str,
    expected_checkpoint_sha256: str,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != expected_status:
        raise ValueError(f"unexpected feature cache report status: {report_path}")
    if report.get("checkpoint_sha256") != expected_checkpoint_sha256:
        raise ValueError(f"feature cache checkpoint identity mismatch: {report_path}")
    if (
        expected_status == "complete_frozen_vjepa_video_feature_cache"
        and report.get("upstream_source_commit") != VJEPA_SOURCE_COMMIT
    ):
        raise ValueError("V-JEPA feature cache source revision mismatch")
    if expected_status == "complete_videoprism_frozen_feature_cache" and (
        report.get("model_revision") != VIDEOPRISM_REVISION
        or report.get("upstream_source_commit") != VIDEOPRISM_SOURCE_COMMIT
    ):
        raise ValueError("VideoPrism feature cache source revision mismatch")
    cache_hash = report.get("feature_file_sha256", report.get("feature_cache_sha256"))
    if cache_hash != sha256_file(cache_path):
        raise ValueError(f"feature cache hash mismatch: {cache_path}")
    manifests = report.get("sample_manifest_sha256", {})
    expected_manifests = {
        name: sha256_file(sample_dir / filename)
        for name, filename in (("train", "train.jsonl"), ("validation", "validation.jsonl"))
    }
    if any(manifests.get(key) != value for key, value in expected_manifests.items()):
        raise ValueError(f"feature cache sample manifest mismatch: {report_path}")
    if "fetch_report" in manifests and manifests["fetch_report"] != sha256_file(
        sample_dir / "fetch-report.json"
    ):
        raise ValueError(f"feature cache fetch report mismatch: {report_path}")
    if report.get("validation_ids") != [row["id"] for row in validation]:
        raise ValueError(f"feature cache validation IDs/order mismatch: {report_path}")
    if report.get("train_ids") != [row["id"] for row in train]:
        raise ValueError(f"feature cache train IDs/order mismatch: {report_path}")
    return report


def _load_scorer(path: Path, device: torch.device) -> tuple[CandidateScorer, str]:
    digest = sha256_file(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    first_weight = state.get("network.0.weight")
    if not isinstance(first_weight, torch.Tensor) or first_weight.ndim != 2:
        raise ValueError(f"unexpected readout state dict: {path}")
    scorer = CandidateScorer(int(first_weight.shape[1]))
    scorer.load_state_dict(state, strict=True)
    scorer.eval().to(device)
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)
    return scorer, digest


def _prediction_rows(
    rows: list[dict[str, Any]], logits: list[torch.Tensor], targets: list[int]
) -> list[dict[str, Any]]:
    predictions = []
    for row, scores, target_index in zip(rows, logits, targets, strict=True):
        probabilities = torch.softmax(scores.float(), dim=-1)
        options = list(row["options"])
        predictions.append(
            {
                "id": row["id"],
                "scene_index": int(row["scene_index"]),
                "question_type": row["question_type"],
                "taxonomy": row["taxonomy"],
                "target": options[target_index],
                "prediction": options[int(probabilities.argmax())],
                "options": options,
                "probabilities": probabilities.tolist(),
            }
        )
    return predictions


def _summarize(
    logits: list[torch.Tensor], targets: list[int], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    metrics = _metrics(logits, targets)
    metrics["mean_confidence"] = float(
        torch.cat([torch.softmax(item.float(), dim=-1).max().reshape(1) for item in logits]).mean()
    )
    by_type = {}
    for kind in sorted({str(row["question_type"]) for row in rows}):
        indexes = [index for index, row in enumerate(rows) if row["question_type"] == kind]
        task = _metrics([logits[index] for index in indexes], [targets[index] for index in indexes])
        task["mean_confidence"] = float(
            torch.cat(
                [torch.softmax(logits[index].float(), dim=-1).max().reshape(1) for index in indexes]
            ).mean()
        )
        by_type[kind] = task
    return {"overall": metrics, "by_question_type": by_type}


def _write_predictions(path: Path, predictions: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions),
        encoding="utf-8",
    )


def main() -> None:
    args = _args()
    started_at = datetime.now(UTC).isoformat()
    sample_dir = args.sample_dir.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    paths = {
        "text_model": args.text_model.resolve(strict=True),
        "vjepa_features": args.vjepa_feature_cache.resolve(strict=True),
        "vjepa_cache_report": args.vjepa_cache_report.resolve(strict=True),
        "vjepa_readout": args.vjepa_readout.resolve(strict=True),
        "videoprism_features": args.videoprism_feature_cache.resolve(strict=True),
        "videoprism_cache_report": args.videoprism_cache_report.resolve(strict=True),
        "videoprism_readout": args.videoprism_readout.resolve(strict=True),
        "excluded_scenes": args.excluded_scenes_json.resolve(strict=True),
    }
    train = _load_rows(sample_dir / "train.jsonl")
    validation = _load_rows(sample_dir / "validation.jsonl")
    _validate_splits(sample_dir, train, validation)
    exclusions = json.loads(paths["excluded_scenes"].read_text(encoding="utf-8"))
    _validate_unseen_scenes(validation, exclusions)
    heldout_scenes = {int(row["scene_index"]) for row in validation}

    vjepa_cache_report = _validate_cache_report(
        paths["vjepa_cache_report"],
        paths["vjepa_features"],
        sample_dir,
        train,
        validation,
        "complete_frozen_vjepa_video_feature_cache",
        VJEPA_CHECKPOINT_SHA256,
    )
    videoprism_cache_report = _validate_cache_report(
        paths["videoprism_cache_report"],
        paths["videoprism_features"],
        sample_dir,
        train,
        validation,
        "complete_videoprism_frozen_feature_cache",
        VIDEOPRISM_CHECKPOINT_SHA256,
    )
    sample_report = json.loads((sample_dir / "fetch-report.json").read_text(encoding="utf-8"))
    if sample_report.get("revision") != ("98b842082ba4f7c18b6b9e3f39145871782a65ef"):
        raise ValueError("CLEVRER sample revision differs from the pinned source")
    expected_media = {str(row["media_sha256"]) for row in train + validation}
    vjepa_features = _feature_map(paths["vjepa_features"])
    videoprism_features = _feature_map(paths["videoprism_features"])
    if set(vjepa_features) != expected_media or set(videoprism_features) != expected_media:
        raise ValueError("video feature media identities differ from the split manifests")
    validation_vjepa = torch.from_numpy(
        np.stack([vjepa_features[str(row["media_sha256"])] for row in validation]).copy()
    )
    validation_videoprism = torch.from_numpy(
        np.stack([videoprism_features[str(row["media_sha256"])] for row in validation]).copy()
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for readout evaluation but is unavailable")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    tokenizer = AutoTokenizer.from_pretrained(paths["text_model"], local_files_only=True)
    text_encoder = (
        AutoModel.from_pretrained(paths["text_model"], local_files_only=True).eval().to(device)
    )
    for parameter in text_encoder.parameters():
        parameter.requires_grad_(False)
    text_features, targets = _text_features(validation, tokenizer, text_encoder, device)
    del text_encoder, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    vjepa_readout, vjepa_readout_sha256 = _load_scorer(paths["vjepa_readout"], device)
    videoprism_readout, videoprism_readout_sha256 = _load_scorer(
        paths["videoprism_readout"], device
    )
    if vjepa_readout_sha256 != VJEPA_READOUT_SHA256:
        raise ValueError("V-JEPA readout is not the previously selected frozen checkpoint")
    if videoprism_readout_sha256 != VIDEOPRISM_READOUT_SHA256:
        raise ValueError("VideoPrism readout is not the previously selected frozen checkpoint")
    started = time.perf_counter()
    vjepa_logits = _predict(vjepa_readout, _combine(validation_vjepa, text_features), device)
    videoprism_logits = _predict(
        videoprism_readout, _combine(validation_videoprism, text_features), device
    )
    evaluation_seconds = time.perf_counter() - started
    vjepa_summary = _summarize(vjepa_logits, targets, validation)
    videoprism_summary = _summarize(videoprism_logits, targets, validation)
    vjepa_predictions = _prediction_rows(validation, vjepa_logits, targets)
    videoprism_predictions = _prediction_rows(validation, videoprism_logits, targets)
    if [row["id"] for row in vjepa_predictions] != [row["id"] for row in videoprism_predictions]:
        raise AssertionError("candidate predictions are not aligned by question ID")

    output.mkdir(parents=True, exist_ok=False)
    vjepa_predictions_path = output / "vjepa-predictions.jsonl"
    videoprism_predictions_path = output / "videoprism-predictions.jsonl"
    _write_predictions(vjepa_predictions_path, vjepa_predictions)
    _write_predictions(videoprism_predictions_path, videoprism_predictions)
    report = {
        "schema_version": 1,
        "status": "complete_frozen_video_readout_evaluation",
        "claim_scope": (
            "held-out CLEVRER scene development diagnostic; fixed previously selected readouts; "
            "not blind, not sealed audit, and not general video understanding"
        ),
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "evaluation_script_sha256": sha256_file(Path(__file__).resolve()),
        "sample_manifest_sha256": {
            name: sha256_file(sample_dir / filename)
            for name, filename in (
                ("train", "train.jsonl"),
                ("validation", "validation.jsonl"),
                ("fetch_report", "fetch-report.json"),
            )
        },
        "excluded_scenes_sha256": sha256_file(paths["excluded_scenes"]),
        "validation_count": len(validation),
        "validation_scene_count": len(heldout_scenes),
        "sampling_seed": sample_report.get("seed"),
        "split_selection_method": (
            "seeded SHA-256 scene/question ordering; previously used scene IDs excluded"
        ),
        "split_witness_only": {
            "train_question_count": len(train),
            "train_scene_count": len({int(row["scene_index"]) for row in train}),
            "use": ("split-integrity witness only; never used to fit or select a readout"),
        },
        "validation_ids_sha256": hashlib.sha256(
            "\n".join(row["id"] for row in validation).encode()
        ).hexdigest(),
        "validation_scene_overlap_with_prior_samples": 0,
        "question_type_counts": {
            kind: sum(row["question_type"] == kind for row in validation)
            for kind in sorted({row["question_type"] for row in validation})
        },
        "readouts": {
            VJEPA_READOUT_ID: {
                "readout_path": str(paths["vjepa_readout"]),
                "readout_sha256": vjepa_readout_sha256,
                "base_checkpoint_sha256": VJEPA_CHECKPOINT_SHA256,
                "base_source_commit": VJEPA_SOURCE_COMMIT,
                "feature_cache_sha256": sha256_file(paths["vjepa_features"]),
                "feature_cache_report_sha256": sha256_file(paths["vjepa_cache_report"]),
                "metrics": vjepa_summary,
                "predictions_file": vjepa_predictions_path.name,
                "predictions_sha256": sha256_file(vjepa_predictions_path),
            },
            VIDEOPRISM_READOUT_ID: {
                "readout_path": str(paths["videoprism_readout"]),
                "readout_sha256": videoprism_readout_sha256,
                "model_revision": VIDEOPRISM_REVISION,
                "base_checkpoint_sha256": VIDEOPRISM_CHECKPOINT_SHA256,
                "base_source_commit": VIDEOPRISM_SOURCE_COMMIT,
                "feature_cache_sha256": sha256_file(paths["videoprism_features"]),
                "feature_cache_report_sha256": sha256_file(paths["videoprism_cache_report"]),
                "metrics": videoprism_summary,
                "predictions_file": videoprism_predictions_path.name,
                "predictions_sha256": sha256_file(videoprism_predictions_path),
            },
        },
        "text_encoder": {
            "revision": TEXT_REVISION,
            "model_path": str(paths["text_model"]),
        },
        "runtime": {
            "device": str(device),
            "evaluation_seconds": evaluation_seconds,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
            "gradient_computation": False,
            "readout_fitting": False,
            "checkpoint_selection": False,
        },
        "feature_cache_reports": {
            "vjepa": vjepa_cache_report,
            "videoprism": videoprism_cache_report,
        },
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "validation_count": len(validation),
                "readouts": {key: value["metrics"] for key, value in report["readouts"].items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
