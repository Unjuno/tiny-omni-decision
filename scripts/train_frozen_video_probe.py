"""Train a small CLEVRER readout over frozen V-JEPA video features."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import random
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from scripts.train_frozen_image_probe import _load_vjepa
from scripts.train_frozen_text_probe import CandidateScorer, _metrics
from tiny_omni_decision.dataset import sha256_file

TEXT_REVISION = "4ca70771034acceecb2e72475f72050fcdde4ddc"
FRAME_COUNT = 8


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--vjepa-source", type=Path, required=True)
    parser.add_argument("--vjepa-checkpoint", type=Path, required=True)
    parser.add_argument("--vjepa-sha256", required=True)
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument("--text-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/pretrained_reuse/path_a_video_vjepa.yaml")
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-questions", type=int, default=8)
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _uniform_indices(frame_count: int, requested: int = FRAME_COUNT) -> list[int]:
    if frame_count < 1 or requested < 1:
        raise ValueError("frame counts must be positive")
    return np.linspace(0, frame_count - 1, num=requested).round().astype(int).tolist()


def _decode_video(path: Path, requested: int = FRAME_COUNT) -> np.ndarray:
    with av.open(str(path)) as container:
        streams = [stream for stream in container.streams if stream.type == "video"]
        if not streams:
            raise ValueError(f"video has no video stream: {path}")
        stream = streams[0]
        total = int(stream.frames or 0)
        if total > 0:
            indices = set(_uniform_indices(total, requested))
            selected: list[np.ndarray] = []
            for index, frame in enumerate(container.decode(stream)):
                if index in indices:
                    selected.append(frame.to_ndarray(format="rgb24"))
                if index >= max(indices):
                    break
            if len(selected) != requested:
                raise ValueError(
                    f"decoder returned {len(selected)}/{requested} selected frames: {path}"
                )
        else:
            decoded = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
            if not decoded:
                raise ValueError(f"video contains no decodable frames: {path}")
            selected = [decoded[index] for index in _uniform_indices(len(decoded), requested)]
    return np.stack(selected)


def _validate_splits(root: Path, train: list[dict[str, Any]], val: list[dict[str, Any]]) -> None:
    for split, rows in (("train", train), ("validation", val)):
        if not rows or any(row.get("split") != split for row in rows):
            raise ValueError(f"empty or mismarked CLEVRER {split} split")
        scene_assets: dict[int, tuple[str, str]] = {}
        for row in rows:
            scene = int(row["scene_index"])
            asset = (str(row["media_sha256"]), str(row["media_path"]))
            if scene in scene_assets and scene_assets[scene] != asset:
                raise ValueError(f"inconsistent media identity for CLEVRER scene {scene}")
            scene_assets[scene] = asset
            media = root / str(row["media_path"])
            if not media.is_file() or sha256_file(media) != row["media_sha256"]:
                raise ValueError(f"missing or hash-mismatched CLEVRER video: {media}")
    if {int(row["scene_index"]) for row in train} & {int(row["scene_index"]) for row in val}:
        raise ValueError("CLEVRER scene overlap between train and validation")
    if {row["media_sha256"] for row in train} & {row["media_sha256"] for row in val}:
        raise ValueError("CLEVRER media overlap between train and validation")
    train_content = {
        (row["question"].casefold().strip(), tuple(sorted(row["options"]))) for row in train
    }
    val_content = {
        (row["question"].casefold().strip(), tuple(sorted(row["options"]))) for row in val
    }
    if train_content & val_content:
        raise ValueError("normalized CLEVRER question content overlap")


def _video_features(
    rows: list[dict[str, Any]], root: Path, encoder: nn.Module, transform: Any, device: torch.device
) -> tuple[Tensor, float, int, dict[str, Tensor]]:
    feature_by_hash: dict[str, Tensor] = {}
    started = time.perf_counter()
    for row in rows:
        if row["media_sha256"] in feature_by_hash:
            continue
        frames = _decode_video(root / row["media_path"])
        clip = transform(frames)[0].unsqueeze(0).to(device=device, dtype=torch.float32)
        if clip.shape[2] != FRAME_COUNT:
            raise ValueError(f"video preprocessing changed the fixed {FRAME_COUNT}-frame input")
        with torch.inference_mode():
            tokens = encoder(clip)
            pooled = tokens.float().mean(dim=1).squeeze(0)
        if pooled.shape != (768,) or not torch.isfinite(pooled).all():
            raise ValueError("V-JEPA produced invalid CLEVRER video features")
        feature_by_hash[str(row["media_sha256"])] = pooled.cpu()
    features = torch.stack([feature_by_hash[str(row["media_sha256"])] for row in rows])
    return features, time.perf_counter() - started, len(feature_by_hash), feature_by_hash


def _text_features(
    rows: list[dict[str, Any]], tokenizer: Any, encoder: nn.Module, device: torch.device
) -> tuple[list[Tensor], list[int]]:
    prompts = [f"Question: {row['question']}" for row in rows]
    prompt_to_options: list[list[str]] = []
    all_options: list[str] = []
    targets = []
    for row in rows:
        options = list(row["options"])
        prompt_to_options.append(options)
        all_options.extend(f"Candidate answer: {option}" for option in options)
        targets.append(options.index(row["target"]))
    texts = prompts + all_options
    vectors: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(texts), 64):
            encoded = tokenizer(
                texts[start : start + 64],
                max_length=128,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            hidden = encoder(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            vectors.extend(((hidden * mask).sum(1) / mask.sum(1).clamp_min(1)).float().cpu())
    question_vectors = vectors[: len(rows)]
    option_vectors = vectors[len(rows) :]
    features = []
    offset = 0
    for index, options in enumerate(prompt_to_options):
        question = question_vectors[index]
        current = []
        for option in option_vectors[offset : offset + len(options)]:
            current.append(
                torch.cat([question, option, (question - option).abs(), question * option])
            )
        features.append(torch.stack(current))
        offset += len(options)
    return features, targets


def _combine(video: Tensor, text: list[Tensor]) -> list[Tensor]:
    return [
        torch.cat([video[index].expand(len(options), -1), options], dim=-1)
        for index, options in enumerate(text)
    ]


def _predict(scorer: nn.Module, features: list[Tensor], device: torch.device) -> list[Tensor]:
    scorer.eval()
    with torch.inference_mode():
        widths = [item.shape[0] for item in features]
        scores = scorer(torch.cat(features).to(device))
    output = []
    offset = 0
    for width in widths:
        output.append(scores[offset : offset + width].float().cpu())
        offset += width
    return output


def main() -> None:
    args = _args()
    run_started_at = datetime.now(UTC).isoformat()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if args.text_revision != TEXT_REVISION:
        raise ValueError(f"unexpected MiniLM revision: {args.text_revision}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("the pinned frozen V-JEPA video probe requires local CUDA")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(4)
    root = args.sample_dir.resolve()
    train = _load_rows(root / "train.jsonl")
    val = _load_rows(root / "validation.jsonl")
    _validate_splits(root, train, val)
    encoder, transform, source_commit = _load_vjepa(args, torch)
    encoder.to(device)
    video_train, train_video_seconds, train_video_encoder_calls, train_feature_cache = (
        _video_features(train, root, encoder, transform, device)
    )
    video_val, val_video_seconds, val_video_encoder_calls, val_feature_cache = _video_features(
        val, root, encoder, transform, device
    )
    del encoder
    torch.cuda.empty_cache()
    feature_cache = output / "video-features.pt"
    torch.save(
        {
            "source_revision": "MIT-IBM/CLEVRER@98b842082ba4f7c18b6b9e3f39145871782a65ef",
            "vjepa_checkpoint_sha256": args.vjepa_sha256,
            "frame_count": FRAME_COUNT,
            "train_ids": [row["id"] for row in train],
            "validation_ids": [row["id"] for row in val],
            "train_scene_ids": [row["scene_index"] for row in train],
            "validation_scene_ids": [row["scene_index"] for row in val],
            "train_media_hashes": [row["media_sha256"] for row in train],
            "validation_media_hashes": [row["media_sha256"] for row in val],
            "train_video_features_by_media_sha256": train_feature_cache,
            "validation_video_features_by_media_sha256": val_feature_cache,
        },
        feature_cache,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.text_model, local_files_only=True)
    text_encoder = (
        AutoModel.from_pretrained(args.text_model, local_files_only=True).eval().to(device)
    )
    for parameter in text_encoder.parameters():
        parameter.requires_grad_(False)
    train_text, train_targets = _text_features(train, tokenizer, text_encoder, device)
    val_text, val_targets = _text_features(val, tokenizer, text_encoder, device)
    train_features = _combine(video_train, train_text)
    val_features = _combine(video_val, val_text)
    del text_encoder, tokenizer
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
        val_logits = _predict(scorer, val_features, device)
        metrics = _metrics(val_logits, val_targets)
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
        raise RuntimeError("validation did not select a video readout checkpoint")
    scorer.load_state_dict(best_state)
    train_inference_started = time.perf_counter()
    train_logits = _predict(scorer, train_features, device)
    torch.cuda.synchronize(device)
    train_inference_seconds = time.perf_counter() - train_inference_started
    validation_inference_started = time.perf_counter()
    val_logits = _predict(scorer, val_features, device)
    torch.cuda.synchronize(device)
    validation_inference_seconds = time.perf_counter() - validation_inference_started
    selected_metrics = _metrics(val_logits, val_targets)
    torch.save(best_state, output / "best-readout.pt")
    predictions = []
    for row, logits, target in zip(val, val_logits, val_targets, strict=True):
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
    (output / "validation-predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions), encoding="utf-8"
    )
    per_type = {}
    for task in sorted({row["question_type"] for row in val}):
        indices = [index for index, row in enumerate(val) if row["question_type"] == task]
        per_type[task] = {
            "count": len(indices),
            **_metrics(
                [val_logits[index] for index in indices], [val_targets[index] for index in indices]
            ),
        }
    report = {
        "schema_version": 1,
        "status": "complete_clevrer_frozen_video_probe",
        "claim_scope": "small synthetic CLEVRER descriptive probe; not general video understanding",
        "seed": args.seed,
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "training_script_sha256": sha256_file(Path(__file__).resolve()),
        "sample_manifest_sha256": {
            "train": sha256_file(root / "train.jsonl"),
            "validation": sha256_file(root / "validation.jsonl"),
            "fetch_report": sha256_file(root / "fetch-report.json"),
        },
        "effective_config": {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_questions": args.batch_questions,
            "frame_count": FRAME_COUNT,
            "video_sample_dir": str(root),
            "vjepa_source_commit": source_commit,
            "vjepa_checkpoint_sha256": args.vjepa_sha256,
            "text_model_revision": args.text_revision,
            "vision_encoder_frozen": True,
            "text_encoder_frozen": True,
        },
        "run_started_utc": run_started_at,
        "run_finished_utc": datetime.now(UTC).isoformat(),
        "frame_count": FRAME_COUNT,
        "training_questions": len(train),
        "validation_questions": len(val),
        "unique_train_scenes": len({row["scene_index"] for row in train}),
        "unique_validation_scenes": len({row["scene_index"] for row in val}),
        "video_encoder_executions": {
            "train": train_video_encoder_calls,
            "validation": val_video_encoder_calls,
        },
        "video_encoder_executions_avoided_by_cache": {
            "train": len(train) - train_video_encoder_calls,
            "validation": len(val) - val_video_encoder_calls,
        },
        "pooled_video_feature_cache_bytes": {
            "train": train_video_encoder_calls * 768 * 4,
            "validation": val_video_encoder_calls * 768 * 4,
            "file": feature_cache.stat().st_size,
            "sha256": sha256_file(feature_cache),
        },
        "train_question_type_counts": {
            task: sum(row["question_type"] == task for row in train)
            for task in ("temporal_descriptive", "static_descriptive")
        },
        "validation_question_type_counts": {
            task: sum(row["question_type"] == task for row in val)
            for task in ("temporal_descriptive", "static_descriptive")
        },
        "validation_metrics": selected_metrics,
        "train_metrics_at_selected_checkpoint": _metrics(train_logits, train_targets),
        "validation_by_question_type": per_type,
        "best_epoch_by_validation_nll": best_epoch,
        "history": history,
        "timing_seconds": {
            "train_video_encoding": train_video_seconds,
            "validation_video_encoding": val_video_seconds,
            "readout_training": training_seconds,
            "train_readout_inference": train_inference_seconds,
            "validation_readout_inference": validation_inference_seconds,
        },
        "validation_readout_ms_per_question": (1000 * validation_inference_seconds / len(val)),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "hardware": {
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            "cuda_runtime": torch.version.cuda,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": importlib.metadata.version("torchvision"),
            "timm": importlib.metadata.version("timm"),
            "av": av.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "attention_backend": "PyTorch SDPA enabled by V-JEPA factory",
        },
        "vjepa_source_commit": source_commit,
        "vjepa_checkpoint_sha256": args.vjepa_sha256,
        "text_model_revision": args.text_revision,
        "train_media_hashes": sorted(row["media_sha256"] for row in train),
        "validation_media_hashes": sorted(row["media_sha256"] for row in val),
        "validation_scene_overlap": 0,
        "trainable_parameters": sum(parameter.numel() for parameter in scorer.parameters()),
        "validation_predictions_sha256": sha256_file(output / "validation-predictions.jsonl"),
        "readout_sha256": sha256_file(output / "best-readout.pt"),
        "artifact_hashes": {
            name: sha256_file(output / name)
            for name in ("video-features.pt", "best-readout.pt", "validation-predictions.jsonl")
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "validation_metrics",
                    "validation_by_question_type",
                    "best_epoch_by_validation_nll",
                    "timing_seconds",
                    "peak_cuda_allocated_bytes",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
