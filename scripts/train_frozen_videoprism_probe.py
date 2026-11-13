"""Train a tiny CLEVRER decision readout from pinned VideoPrism features."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from scripts.audit_videoprism_checkpoint import MODEL_REVISION
from scripts.train_frozen_text_probe import CandidateScorer, _metrics
from scripts.train_frozen_video_probe import (
    _combine,
    _load_rows,
    _predict,
    _text_features,
    _validate_splits,
)

TEXT_REVISION = "4ca70771034acceecb2e72475f72050fcdde4ddc"
CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--cache-report", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument("--text-revision", default=TEXT_REVISION)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/pretrained_reuse/path_a_video_videoprism.yaml")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-questions", type=int, default=8)
    return parser.parse_args()


def _load_features(
    cache_path: Path,
    report_path: Path,
    sample_dir: Path,
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "complete_videoprism_frozen_feature_cache":
        raise ValueError("feature cache is not a completed VideoPrism extraction")
    if report.get("model_revision") != MODEL_REVISION:
        raise ValueError("feature cache model revision mismatch")
    if report.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("feature cache checkpoint SHA-256 mismatch")
    if report.get("sample_manifest_sha256") != {
        "train": sha256_file(sample_dir / "train.jsonl"),
        "validation": sha256_file(sample_dir / "validation.jsonl"),
        "fetch_report": sha256_file(sample_dir / "fetch-report.json"),
    }:
        raise ValueError("feature cache sample manifest hashes differ")
    if report.get("train_ids") != [row["id"] for row in train] or report.get("validation_ids") != [
        row["id"] for row in validation
    ]:
        raise ValueError("feature cache sample IDs or ordering differ from the probe")
    if report.get("feature_cache_sha256") != sha256_file(cache_path):
        raise ValueError("feature cache file hash mismatch")

    with np.load(cache_path, allow_pickle=False) as payload:
        media_hashes = payload["media_sha256"].astype(str).tolist()
        values = payload["features"]
    if values.shape != (len(media_hashes), 768) or not np.isfinite(values).all():
        raise ValueError("pooled feature tensor has invalid shape or values")
    features_by_media = dict(zip(media_hashes, values, strict=True))
    expected_media = {row["media_sha256"] for row in train + validation}
    if set(features_by_media) != expected_media:
        raise ValueError("feature media identities differ from the split manifests")
    train_features = torch.from_numpy(
        np.stack([features_by_media[row["media_sha256"]] for row in train]).copy()
    )
    validation_features = torch.from_numpy(
        np.stack([features_by_media[row["media_sha256"]] for row in validation]).copy()
    )
    return train_features, validation_features


def main() -> None:
    args = _args()
    run_started = datetime.now(UTC).isoformat()
    sample_dir = args.sample_dir.resolve(strict=True)
    cache_path = args.feature_cache.resolve(strict=True)
    cache_report_path = args.cache_report.resolve(strict=True)
    text_model_path = args.text_model.resolve(strict=True)
    config_path = args.config.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    if args.text_revision != TEXT_REVISION:
        raise ValueError(f"unexpected MiniLM revision: {args.text_revision}")
    torch.manual_seed(args.seed)
    train = _load_rows(sample_dir / "train.jsonl")
    validation = _load_rows(sample_dir / "validation.jsonl")
    _validate_splits(sample_dir, train, validation)
    video_train, video_validation = _load_features(
        cache_path, cache_report_path, sample_dir, train, validation
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.set_num_threads(4)

    tokenizer = AutoTokenizer.from_pretrained(text_model_path, local_files_only=True)
    text_encoder = (
        AutoModel.from_pretrained(text_model_path, local_files_only=True).eval().to(device)
    )
    for parameter in text_encoder.parameters():
        parameter.requires_grad_(False)
    train_text, train_targets = _text_features(train, tokenizer, text_encoder, device)
    validation_text, validation_targets = _text_features(
        validation, tokenizer, text_encoder, device
    )
    train_features = _combine(video_train, train_text)
    validation_features = _combine(video_validation, validation_text)
    del text_encoder, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    scorer = CandidateScorer(train_features[0].shape[-1]).to(device)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=1e-3, weight_decay=1e-3)
    history = []
    best_state = None
    best_nll = float("inf")
    best_epoch = 0
    started = time.perf_counter()
    generator = torch.Generator().manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        scorer.train()
        order = torch.randperm(len(train_features), generator=generator).tolist()
        losses = []
        for start in range(0, len(order), args.batch_questions):
            indices = order[start : start + args.batch_questions]
            rows = [train_features[index].to(device) for index in indices]
            targets = torch.tensor([train_targets[index] for index in indices], device=device)
            widths = [row.shape[0] for row in rows]
            scores = scorer(torch.cat(rows))
            logits = scores.new_full((len(rows), max(widths)), -1e4)
            offset = 0
            for batch_index, width in enumerate(widths):
                logits[batch_index, :width] = scores[offset : offset + width]
                offset += width
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        val_logits = _predict(scorer, validation_features, device)
        metrics = _metrics(val_logits, validation_targets)
        history.append(
            {"epoch": epoch, "train_ce": sum(losses) / len(losses), "validation": metrics}
        )
        if metrics["nll"] < best_nll:
            best_nll = metrics["nll"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in scorer.state_dict().items()
            }
    training_seconds = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("validation did not select a readout checkpoint")
    scorer.load_state_dict(best_state)
    train_logits = _predict(scorer, train_features, device)
    val_logits = _predict(scorer, validation_features, device)
    selected_metrics = _metrics(val_logits, validation_targets)
    output.mkdir(parents=True, exist_ok=False)
    readout_path = output / "best-readout.pt"
    torch.save(best_state, readout_path)
    predictions = []
    for row, logits, target in zip(validation, val_logits, validation_targets, strict=True):
        probabilities = torch.softmax(logits, dim=-1)
        options = row["options"]
        predictions.append(
            {
                "id": row["id"],
                "scene_index": row["scene_index"],
                "question_type": row["question_type"],
                "taxonomy": row["taxonomy"],
                "target": options[target],
                "prediction": options[int(probabilities.argmax())],
                "options": options,
                "probabilities": probabilities.tolist(),
            }
        )
    predictions_path = output / "validation-predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions),
        encoding="utf-8",
    )
    per_type = {}
    for task in sorted({row["question_type"] for row in validation}):
        indices = [index for index, row in enumerate(validation) if row["question_type"] == task]
        per_type[task] = {
            "count": len(indices),
            **_metrics(
                [val_logits[index] for index in indices],
                [validation_targets[index] for index in indices],
            ),
        }
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    cache_report = json.loads(cache_report_path.read_text(encoding="utf-8"))
    report = {
        "schema_version": 1,
        "status": "complete_clevrer_frozen_videoprism_probe",
        "claim_scope": "small synthetic CLEVRER descriptive probe; not general video understanding",
        "seed": args.seed,
        "source_commit": source_commit,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "training_script_sha256": sha256_file(Path(__file__).resolve()),
        "feature_cache_sha256": sha256_file(cache_path),
        "feature_cache_report_sha256": sha256_file(cache_report_path),
        "sample_manifest_sha256": {
            name: sha256_file(sample_dir / filename)
            for name, filename in (
                ("train", "train.jsonl"),
                ("validation", "validation.jsonl"),
                ("fetch_report", "fetch-report.json"),
            )
        },
        "model": {
            "repo_id": "google/videoprism-base-f16r288",
            "revision": MODEL_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "frozen": True,
        },
        "effective_config": {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_questions": args.batch_questions,
            "frame_count": cache_report["frame_count"],
            "resolution": cache_report["frame_size"],
            "sample_dir": str(sample_dir),
            "text_model_revision": args.text_revision,
            "video_encoder_frozen": True,
            "text_encoder_frozen": True,
            "readout_selection": "minimum_validation_nll",
        },
        "run_started_utc": run_started,
        "run_finished_utc": datetime.now(UTC).isoformat(),
        "training_questions": len(train),
        "validation_questions": len(validation),
        "unique_train_scenes": len({row["scene_index"] for row in train}),
        "unique_validation_scenes": len({row["scene_index"] for row in validation}),
        "unique_video_features": cache_report["unique_video_count"],
        "train_question_type_counts": {
            kind: sum(row["question_type"] == kind for row in train)
            for kind in ("temporal_descriptive", "static_descriptive")
        },
        "validation_question_type_counts": {
            kind: sum(row["question_type"] == kind for row in validation)
            for kind in ("temporal_descriptive", "static_descriptive")
        },
        "validation_metrics": selected_metrics,
        "train_metrics_at_selected_checkpoint": _metrics(
            train_logits, [row["options"].index(row["target"]) for row in train]
        ),
        "validation_by_question_type": per_type,
        "best_epoch_by_validation_nll": best_epoch,
        "history": history,
        "training_seconds": training_seconds,
        "device": str(device),
        "hardware": {
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory
            if device.type == "cuda"
            else None,
            "cuda_runtime": torch.version.cuda if device.type == "cuda" else None,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": importlib.metadata.version("torchvision"),
            "av": importlib.metadata.version("av"),
            "transformers": importlib.metadata.version("transformers"),
        },
        "train_media_hashes": sorted({row["media_sha256"] for row in train}),
        "validation_media_hashes": sorted({row["media_sha256"] for row in validation}),
        "validation_scene_overlap": 0,
        "trainable_parameters": sum(parameter.numel() for parameter in scorer.parameters()),
        "validation_predictions_sha256": sha256_file(predictions_path),
        "readout_sha256": sha256_file(readout_path),
        "artifact_hashes": {
            name: sha256_file(output / name)
            for name in ("best-readout.pt", "validation-predictions.jsonl")
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "validation_metrics": selected_metrics,
                "validation_by_question_type": per_type,
                "best_epoch_by_validation_nll": best_epoch,
                "training_seconds": training_seconds,
                "device": str(device),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
