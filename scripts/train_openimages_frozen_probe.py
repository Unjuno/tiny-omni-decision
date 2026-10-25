"""Train a small option scorer over frozen VideoPrism and MiniLM features."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]

TEXT_REVISION = "4ca70771034acceecb2e72475f72050fcdde4ddc"
TEXT_WEIGHTS_SHA256 = "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"
IMAGE_CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
IMAGE_SIZE = 768
TEXT_SIZE = 384
HIDDEN_SIZE = 128
TRAINABLE_PARAMETERS = 147_713


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _load_features(features_dir: Path, split: str) -> tuple[list[dict[str, Any]], np.ndarray]:
    rows = _read_rows(features_dir / f"{split}-rows.jsonl")
    with np.load(features_dir / f"{split}-features.npz", allow_pickle=False) as data:
        sample_ids = data["sample_ids"].astype(str).tolist()
        image_ids = data["image_ids"].astype(str).tolist()
        features = data["features"].astype(np.float32, copy=True)
        media_hashes = data["media_sha256"].astype(str).tolist()
    if [str(row["sample_id"]) for row in rows] != sample_ids:
        raise ValueError(f"{split} row order differs from feature sample IDs")
    if [str(row["image_id"]) for row in rows] != image_ids:
        raise ValueError(f"{split} row order differs from feature image IDs")
    if [str(row["media_sha256"]) for row in rows] != media_hashes:
        raise ValueError(f"{split} row order differs from feature media hashes")
    if features.shape != (len(rows), IMAGE_SIZE) or not np.isfinite(features).all():
        raise ValueError(f"invalid {split} frozen image features")
    return rows, features


def _validate_rows(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> None:
    for name, rows in (("train", train_rows), ("validation", val_rows)):
        if any(len(row["options"]) != 4 or len(set(row["options"])) != 4 for row in rows):
            raise ValueError(f"{name} examples must have exactly four unique options")
        if any(row["target"] not in row["options"] for row in rows):
            raise ValueError(f"{name} target is not among its options")
    for key in ("image_id", "media_sha256", "normalized_rgb_sha256", "original_md5_base64"):
        if {str(row[key]) for row in train_rows} & {str(row[key]) for row in val_rows}:
            raise ValueError(f"train/validation {key} overlap")


def _text_features(
    rows: list[dict[str, Any]], tokenizer: Any, encoder: nn.Module, batch_size: int = 128
) -> np.ndarray:
    prompt_to_index: dict[str, int] = {}
    prompts = []
    ranges = []
    for row in rows:
        indices = []
        for option in row["options"]:
            prompt = f"Question: {row['question']} Candidate answer: {option}."
            if prompt not in prompt_to_index:
                prompt_to_index[prompt] = len(prompts)
                prompts.append(prompt)
            indices.append(prompt_to_index[prompt])
        ranges.append(indices)

    pooled_chunks = []
    encoder.eval()
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            encoded = tokenizer(
                prompts[start : start + batch_size],
                truncation=False,
                padding=True,
                return_tensors="pt",
            )
            if int(encoded["input_ids"].shape[1]) > 96:
                raise ValueError("candidate prompt exceeds frozen text encoder limit")
            hidden = encoder(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = ((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)).float()
            pooled_chunks.append(pooled.cpu())
    unique_features = torch.cat(pooled_chunks)
    values = unique_features[torch.tensor(ranges, dtype=torch.long)].numpy()
    if values.shape != (len(rows), 4, TEXT_SIZE) or not np.isfinite(values).all():
        raise ValueError("invalid MiniLM candidate-text features")
    return values


def _tasks(
    image_features: np.ndarray, text_features: np.ndarray, rows: list[dict[str, Any]]
) -> tuple[Tensor, Tensor]:
    images = torch.from_numpy(image_features).unsqueeze(1).expand(-1, 4, -1)
    texts = torch.from_numpy(text_features)
    features = torch.cat((images, texts), dim=-1)
    targets = torch.tensor([row["options"].index(row["target"]) for row in rows], dtype=torch.long)
    return features, targets


def _quality(logits: Tensor, targets: Tensor) -> dict[str, Any]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    predictions = probabilities.argmax(dim=-1)
    correct = predictions.eq(targets)
    target_probs = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-12)
    nll = -target_probs.log()
    one_hot = F.one_hot(targets, num_classes=probabilities.shape[-1]).to(probabilities.dtype)
    brier = (probabilities - one_hot).square().sum(dim=-1)
    confidence = probabilities.max(dim=-1).values
    ece = 0.0
    bin_counts = []
    for index in range(15):
        low, high = index / 15, (index + 1) / 15
        mask = (confidence >= low) & (confidence < high if index < 14 else confidence <= high)
        count = int(mask.sum())
        bin_counts.append(count)
        if count:
            ece += (
                count
                / len(targets)
                * abs(float(correct[mask].float().mean() - confidence[mask].mean()))
            )
    return {
        "count": int(len(targets)),
        "accuracy": float(correct.float().mean()),
        "nll": float(nll.mean()),
        "brier": float(brier.mean()),
        "ece_15_bins": ece,
        "ece_bin_counts": bin_counts,
        "mean_confidence": float(confidence.mean()),
    }


def _predictions(rows: list[dict[str, Any]], logits: Tensor) -> list[dict[str, Any]]:
    probabilities = torch.softmax(logits.float(), dim=-1).cpu()
    result = []
    for index, row in enumerate(rows):
        result.append(
            {
                "sample_id": row["sample_id"],
                "image_id": row["image_id"],
                "target_label": row["target_label"],
                "target": row["target"],
                "options": row["options"],
                "prediction": row["options"][int(probabilities[index].argmax())],
                "probabilities": probabilities[index].tolist(),
            }
        )
    return result


def _write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument("--image-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pretrained_reuse/path_a_openimages_videoprism.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-questions", type=int, default=64)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    features_dir = args.features_dir.resolve(strict=True)
    text_model = args.text_model.resolve(strict=True)
    image_checkpoint = args.image_checkpoint.resolve(strict=True)
    config_path = args.config.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("model/readout artifacts must be stored outside the repository")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite training output: {output}")
    if args.epochs < 1 or args.epochs > 24 or args.batch_questions < 1:
        raise ValueError("epochs must be 1..24 and batch_questions must be positive")
    if _sha256(text_model / "model.safetensors") != TEXT_WEIGHTS_SHA256:
        raise ValueError("MiniLM weights differ from the pinned revision")
    if _sha256(image_checkpoint) != IMAGE_CHECKPOINT_SHA256:
        raise ValueError("VideoPrism checkpoint differs from the pinned revision")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("experiment_id") != "path-a-openimages-videoprism-minilm-seed17":
        raise ValueError("unexpected Open Images experiment config")
    if not args.smoke and (args.epochs != 8 or args.batch_questions != 64):
        raise ValueError("full run must use the frozen eight-epoch config")

    run_started_utc = datetime.now(UTC).isoformat()
    train_rows, train_images = _load_features(features_dir, "train")
    val_rows, val_images = _load_features(features_dir, "validation")
    if args.smoke:
        train_rows, train_images = train_rows[:8], train_images[:8]
        val_rows, val_images = val_rows[:8], val_images[:8]
        args.epochs = 1
    _validate_rows(train_rows, val_rows)

    torch.set_num_threads(4)
    torch.manual_seed(17)
    random.seed(17)
    np.random.seed(17)
    device = torch.device("cpu")
    text_weights = text_model / "model.safetensors"
    tokenizer = AutoTokenizer.from_pretrained(text_model, local_files_only=True)
    encoder = AutoModel.from_pretrained(text_model, local_files_only=True).eval()
    encoder.requires_grad_(False)
    text_parameter_count = sum(parameter.numel() for parameter in encoder.parameters())
    if text_parameter_count != 17_389_824:
        raise ValueError(f"unexpected MiniLM parameter count: {text_parameter_count}")

    extraction_started = time.perf_counter()
    train_text = _text_features(train_rows, tokenizer, encoder)
    val_text = _text_features(val_rows, tokenizer, encoder)
    text_feature_seconds = time.perf_counter() - extraction_started
    train_x, train_y = _tasks(train_images, train_text, train_rows)
    val_x, val_y = _tasks(val_images, val_text, val_rows)
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise ValueError("frozen MiniLM unexpectedly has trainable parameters")
    del encoder, tokenizer

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    from tiny_omni_decision.decision import FrozenFeatureCandidateScorer

    scorer = FrozenFeatureCandidateScorer(IMAGE_SIZE + TEXT_SIZE, hidden_size=HIDDEN_SIZE).to(
        device
    )
    parameter_count = sum(parameter.numel() for parameter in scorer.parameters())
    if parameter_count != TRAINABLE_PARAMETERS:
        raise ValueError(f"unexpected readout parameter count: {parameter_count}")
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=1e-3, weight_decay=1e-4)
    order_rng = random.Random(17)
    history = []
    best_nll = float("inf")
    best_epoch = 0
    best_state = None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_rows)))
        order_rng.shuffle(order)
        scorer.train()
        losses = []
        for start in range(0, len(order), args.batch_questions):
            indices = order[start : start + args.batch_questions]
            batch_x = train_x[indices]
            batch_y = train_y[indices]
            logits = scorer(batch_x)
            loss = F.cross_entropy(logits, batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        scorer.eval()
        with torch.inference_mode():
            val_logits = scorer(val_x)
            train_logits = scorer(train_x)
        val_metrics = _quality(val_logits, val_y)
        history.append(
            {
                "epoch": epoch,
                "train_ce": float(np.mean(losses)),
                "train_metrics": _quality(train_logits, train_y),
                "validation_metrics": val_metrics,
            }
        )
        print(json.dumps(history[-1], sort_keys=True), flush=True)
        if val_metrics["nll"] < best_nll:
            best_nll = val_metrics["nll"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in scorer.state_dict().items()
            }
    training_seconds = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("validation checkpoint selection did not produce a checkpoint")
    final_state = {key: value.detach().cpu().clone() for key, value in scorer.state_dict().items()}

    output.mkdir(parents=True, exist_ok=False)
    scorer.load_state_dict(best_state)
    scorer.eval()
    with torch.inference_mode():
        best_logits_before_reload = scorer(val_x).detach().clone()
    torch.save(best_state, output / "best-readout.pt")
    torch.save(final_state, output / "final-readout.pt")
    scorer.load_state_dict(
        torch.load(output / "best-readout.pt", map_location=device, weights_only=True)
    )
    with torch.inference_mode():
        reloaded_logits = scorer(val_x)
    reload_delta = float((reloaded_logits - best_logits_before_reload).abs().max())
    selected_metrics = _quality(reloaded_logits, val_y)
    validation_by_target = {}
    for label in sorted({str(row["target_label"]) for row in val_rows}):
        indices = [index for index, row in enumerate(val_rows) if str(row["target_label"]) == label]
        validation_by_target[label] = _quality(reloaded_logits[indices], val_y[indices])
    predictions_path = output / "validation-predictions.jsonl"
    predictions_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in _predictions(val_rows, reloaded_logits)
        ),
        encoding="utf-8",
    )
    baseline_logits = torch.zeros((len(val_rows), 4), dtype=torch.float32)
    uniform_metrics = _quality(baseline_logits, val_y)
    majority_index = int(torch.bincount(train_y, minlength=4).argmax())
    position_logits = torch.full((len(val_rows), 4), -20.0)
    position_logits[:, majority_index] = 20.0
    position_metrics = _quality(position_logits, val_y)

    metadata = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "started_utc": run_started_utc,
        "source_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "runner_sha256": _sha256(Path(__file__)),
        "config_sha256": _sha256(config_path),
        "feature_config_sha256": _sha256(features_dir / "config.json"),
        "video_encoder_checkpoint_bytes": image_checkpoint.stat().st_size,
        "feature_hashes": {
            split: _sha256(features_dir / f"{split}-features.npz")
            for split in ("train", "validation")
        },
        "text_encoder": {
            "revision": TEXT_REVISION,
            "weights_sha256": _sha256(text_weights),
            "parameters": text_parameter_count,
            "frozen": True,
        },
        "image_encoder": {
            "revision": "be719a406d563b66f0ac969e7c94bab8e997c81a",
            "parameters": 114_365_184,
            "frozen": True,
        },
        "trainable_readout_parameters": parameter_count,
        "total_parameters": 114_365_184 + text_parameter_count + parameter_count,
        "split_counts": {"train": len(train_rows), "validation": len(val_rows)},
        "target_label_counts": {
            split: dict(sorted(Counter(str(row["target_label"]) for row in rows).items()))
            for split, rows in (("train", train_rows), ("validation", val_rows))
        },
        "split_ids_sha256": {
            "train": hashlib.sha256(
                "\n".join(str(row["sample_id"]) for row in train_rows).encode()
            ).hexdigest(),
            "validation": hashlib.sha256(
                "\n".join(str(row["sample_id"]) for row in val_rows).encode()
            ).hexdigest(),
        },
        "seed": 17,
        "device": "CPU",
        "smoke": args.smoke,
        "epochs_configured": args.epochs,
        "batch_questions": args.batch_questions,
        "checkpoint_selector": "earliest minimum validation NLL",
        "best_epoch": best_epoch,
        "history": history,
        "selected_validation_metrics": selected_metrics,
        "selected_validation_metrics_by_target_label": validation_by_target,
        "uniform_option_baseline": uniform_metrics,
        "training_option_position_majority_baseline": position_metrics,
        "majority_option_index": majority_index,
        "text_feature_seconds": text_feature_seconds,
        "readout_training_seconds": training_seconds,
        "validation_reload_max_logit_delta": reload_delta,
        "sealed_audit_loaded": False,
        "training_split_and_model_inputs": (
            "fixed Open Images V6 hierarchy-choice development pilot"
        ),
        "package_size_bytes": {
            "videoprism_checkpoint": image_checkpoint.stat().st_size,
            "minilm_weights": text_weights.stat().st_size,
            "best_readout": (output / "best-readout.pt").stat().st_size,
            "estimated_combined_runtime_weights": (
                image_checkpoint.stat().st_size
                + text_weights.stat().st_size
                + (output / "best-readout.pt").stat().st_size
            ),
        },
        "artifact_sha256": {
            name: _sha256(output / name)
            for name in ("best-readout.pt", "final-readout.pt", "validation-predictions.jsonl")
        },
        "versions": {
            key: importlib.metadata.version(key) for key in ("torch", "transformers", "numpy")
        },
        "completed_utc": datetime.now(UTC).isoformat(),
    }
    _write(output / "run-metadata.json", metadata)
    _write(output / "selected-validation-metrics.json", selected_metrics)
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
