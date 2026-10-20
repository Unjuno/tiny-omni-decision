"""Fit and evaluate the frozen-feature AV event-source Decision readout."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from scripts.audit_perception_av_correspondence import sha256_file
from tiny_omni_decision.av_event import AVEventSourceScorer

SPLITS = {"fit", "selection", "calibration", "development_validation"}
AUDIO_SIZE = 512
VISUAL_SIZE = 384
DINOV2_TOKENS = 257


def pool_dinov2_patch_tokens(features: np.ndarray) -> np.ndarray:
    """Use the pretrained CLS representation, keeping candidate order intact."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim == 2 and values.shape[1:] == (VISUAL_SIZE,):
        if values.shape[0] < 2 or not np.isfinite(values).all():
            raise ValueError(
                "pooled DINOv2 candidate features must be finite with at least two options"
            )
        return values.copy()
    if values.ndim != 3 or values.shape[1:] != (DINOV2_TOKENS, VISUAL_SIZE):
        raise ValueError(
            "DINOv2 features must have shape [candidates, 384] or [candidates, 257, 384]"
        )
    if values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("DINOv2 candidate features must be finite with at least two options")
    pooled = values[:, 0, :].copy()
    if not np.isfinite(pooled).all():
        raise ValueError("pooled DINOv2 features are non-finite")
    return pooled


def validate_video_splits(rows: list[dict[str, Any]]) -> None:
    """Reject duplicate event IDs and any video assigned to multiple splits."""
    seen_ids: set[str] = set()
    video_splits: dict[str, str] = {}
    media_splits: dict[str, str] = {}
    for row in rows:
        row_id, video_id, split = row.get("id"), row.get("video_id"), row.get("split")
        if not isinstance(row_id, str) or not row_id or row_id in seen_ids:
            raise ValueError("event IDs must be nonempty and unique")
        if not isinstance(video_id, str) or not video_id or split not in SPLITS:
            raise ValueError("each row needs a video ID and recognized split")
        seen_ids.add(row_id)
        previous = video_splits.setdefault(video_id, split)
        if previous != split:
            raise ValueError("video appears in multiple splits")
        media_hash = row.get("media_sha256")
        if media_hash is not None:
            if (
                not isinstance(media_hash, str)
                or len(media_hash) != 64
                or any(char not in "0123456789abcdef" for char in media_hash)
            ):
                raise ValueError("media_sha256 must be a lowercase SHA-256 digest")
            previous_media_split = media_splits.setdefault(media_hash, split)
            if previous_media_split != split:
                raise ValueError("media identity appears in multiple splits")


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("feature manifest is empty")
    validate_video_splits(rows)
    return rows


def _load_array(path_value: str, expected_sha256: str, name: str) -> np.ndarray:
    path = Path(path_value)
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError(f"{name} feature file is missing or hash-mismatched")
    values = np.load(path, allow_pickle=False)
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError(f"{name} feature file must contain finite numeric values")
    return np.asarray(values, dtype=np.float32)


def load_feature_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_rows(path)
    loaded = []
    for row in rows:
        required = (
            "audio_feature_path",
            "audio_feature_sha256",
            "visual_feature_path",
            "visual_feature_sha256",
            "candidate_order_sha256",
            "media_sha256",
            "annotation_sha256",
        )
        missing = [key for key in required if not isinstance(row.get(key), str) or not row[key]]
        if missing:
            raise ValueError(f"feature row is missing required field: {missing[0]}")
        order_hash = row["candidate_order_sha256"]
        if len(order_hash) != 64 or any(char not in "0123456789abcdef" for char in order_hash):
            raise ValueError("candidate_order_sha256 must be a lowercase SHA-256 digest")
        audio = _load_array(row["audio_feature_path"], row["audio_feature_sha256"], "audio")
        visual_tokens = _load_array(
            row["visual_feature_path"], row["visual_feature_sha256"], "visual"
        )
        if audio.shape != (AUDIO_SIZE,):
            raise ValueError("audio event feature must have shape [512]")
        visual = pool_dinov2_patch_tokens(visual_tokens)
        candidate_count = row.get("candidate_count")
        if type(candidate_count) is not int or visual.shape[0] != candidate_count:
            raise ValueError("candidate count differs from visual feature rows")
        target = row.get("target_index")
        if type(target) is not int or not 0 <= target < visual.shape[0]:
            raise ValueError("target index outside candidate list")
        loaded.append({**row, "audio": audio, "visual": visual})
    return loaded


def select_best_epoch(history: list[dict[str, Any]], min_delta: float = 0.0) -> int:
    if not history:
        raise ValueError("selection history is empty")
    if min_delta < 0:
        raise ValueError("min_delta must be nonnegative")
    valid = [row for row in history if math.isfinite(float(row["selection_nll"]))]
    if not valid:
        raise ValueError("selection history has no finite NLL")
    best = valid[0]
    best_nll = float(best["selection_nll"])
    for row in valid[1:]:
        score = float(row["selection_nll"])
        if score < best_nll - min_delta:
            best, best_nll = row, score
    return int(best["epoch"])


def _metric_summary(logits: list[torch.Tensor], targets: list[int]) -> dict[str, float]:
    if not logits or len(logits) != len(targets):
        raise ValueError("metrics require one target for every nonempty event-logit vector")
    probabilities = [item.detach().float().cpu().softmax(dim=-1) for item in logits]
    if any(item.ndim != 1 or item.numel() < 2 for item in probabilities):
        raise ValueError("each event must have logits for at least two candidates")
    if any(
        type(target) is not int or not 0 <= target < item.numel()
        for item, target in zip(probabilities, targets, strict=True)
    ):
        raise ValueError("event target is outside its candidate range")
    confidence = torch.stack([item.max() for item in probabilities])
    prediction = torch.tensor([int(item.argmax()) for item in probabilities], dtype=torch.long)
    target_tensor = torch.tensor(targets, dtype=torch.long)
    correct = prediction.eq(target_tensor)
    nll = torch.stack(
        [
            -torch.log(item[target].clamp_min(torch.finfo(item.dtype).tiny))
            for item, target in zip(probabilities, targets, strict=True)
        ]
    ).mean()
    brier = torch.stack(
        [
            (item - F.one_hot(torch.tensor(target), num_classes=item.numel()).float())
            .square()
            .sum()
            for item, target in zip(probabilities, targets, strict=True)
        ]
    ).mean()
    ece = 0.0
    for lower in torch.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (confidence >= lower) & (
            (confidence < upper) if upper < 1.0 else (confidence <= upper)
        )
        if mask.any():
            ece += float(mask.float().mean()) * abs(
                float(confidence[mask].mean()) - float(correct[mask].float().mean())
            )
    return {
        "accuracy": float(correct.float().mean()),
        "nll": float(nll),
        "brier": float(brier),
        "ece": float(ece),
    }


def _score_rows(model: AVEventSourceScorer, rows: list[dict[str, Any]], condition: str):
    logits: list[torch.Tensor] = []
    targets: list[int] = []
    predictions = []
    model.eval()
    with torch.inference_mode():
        for row in rows:
            kwargs = {"visual_features": torch.from_numpy(row["visual"])}
            if condition == "joint":
                kwargs["audio_features"] = torch.from_numpy(row["audio"])
            score = model(condition=condition, **kwargs).float()
            target = row["target_index"]
            probs = score.softmax(dim=-1).cpu().tolist()
            logits.append(score.cpu())
            targets.append(target)
            predictions.append(
                {
                    "id": row["id"],
                    "video_id": row["video_id"],
                    "target_index": target,
                    "prediction_index": int(np.argmax(probs)),
                    "probabilities": probs,
                    "candidate_count": len(probs),
                    "candidate_order_sha256": row["candidate_order_sha256"],
                }
            )
    return _metric_summary(logits, targets), predictions


def _atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def fit_readout(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    condition: str,
    seed: int = 17,
    epochs: int = 100,
    patience: int = 20,
    learning_rate: float = 1e-3,
    min_delta: float = 1e-4,
    hidden_size: int = 128,
    max_updates: int | None = None,
    evaluation_interval: int | None = None,
) -> dict[str, Any]:
    if condition not in {"joint", "video_only"}:
        raise ValueError("only trainable joint/video_only conditions are supported")
    if Path(output_dir).exists():
        raise ValueError("fresh output directory required")
    if epochs < 1 or patience < 1 or learning_rate <= 0 or min_delta < 0:
        raise ValueError("invalid training schedule")
    if (max_updates is None) != (evaluation_interval is None):
        raise ValueError("max_updates and evaluation_interval must be supplied together")
    if max_updates is not None and (
        max_updates < 1 or evaluation_interval is None or evaluation_interval < 1
    ):
        raise ValueError("fixed-update training requires positive updates and validation interval")
    validate_video_splits(rows)
    fit_rows = [row for row in rows if row["split"] == "fit"]
    selection_rows = [row for row in rows if row["split"] == "selection"]
    if not fit_rows or not selection_rows:
        raise ValueError("fit and video-disjoint selection rows are both required")
    if not any(row["split"] == "calibration" for row in rows) or not any(
        row["split"] == "development_validation" for row in rows
    ):
        raise ValueError("calibration and development-validation splits are both required")
    if {row["video_id"] for row in fit_rows} & {row["video_id"] for row in selection_rows}:
        raise ValueError("fit and selection videos overlap")
    Path(output_dir).mkdir(parents=True, exist_ok=False)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = AVEventSourceScorer(hidden_size=hidden_size)
    active_prefixes = ("visual_projection.", "video_head.")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(condition == "joint" or name.startswith(active_prefixes))
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )
    generator = torch.Generator().manual_seed(seed)
    history: list[dict[str, Any]] = []
    best_nll = float("inf")
    best_epoch = None
    best_state = None
    best_predictions = None
    best_step = None
    stale_epochs = 0
    started = time.perf_counter()
    update_limit = max_updates if max_updates is not None else epochs * len(fit_rows)
    validation_interval = evaluation_interval if evaluation_interval is not None else len(fit_rows)
    optimizer_updates = 0
    interval_loss_sum = 0.0
    interval_loss_count = 0

    def validate(epoch: int) -> None:
        nonlocal best_nll, best_epoch, best_state, best_predictions
        nonlocal best_step, stale_epochs, interval_loss_sum, interval_loss_count
        selection_metrics, predictions = _score_rows(model, selection_rows, condition)
        row_history = {
            "epoch": epoch,
            "optimizer_updates": optimizer_updates,
            "train_ce": interval_loss_sum / interval_loss_count,
            "train_ce_interval_updates": interval_loss_count,
            "selection_nll": selection_metrics["nll"],
            "selection_metrics": selection_metrics,
        }
        history.append(row_history)
        improved = selection_metrics["nll"] < best_nll - min_delta
        if improved:
            best_nll = selection_metrics["nll"]
            best_epoch = epoch
            best_step = optimizer_updates
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            best_predictions = predictions
            stale_epochs = 0
            _atomic_torch_save(
                {
                    "state_dict": best_state,
                    "condition": condition,
                    "seed": seed,
                    "epoch": epoch,
                    "optimizer_updates": optimizer_updates,
                    "hidden_size": hidden_size,
                    "selector": "minimum video-disjoint selection NLL; min_delta applied",
                },
                Path(output_dir) / "best.pt",
            )
        else:
            stale_epochs += 1
        interval_loss_sum = 0.0
        interval_loss_count = 0

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(fit_rows), generator=generator).tolist()
        for index in order:
            if optimizer_updates >= update_limit:
                break
            row = fit_rows[index]
            optimizer.zero_grad(set_to_none=True)
            kwargs = {"visual_features": torch.from_numpy(row["visual"])}
            if condition == "joint":
                kwargs["audio_features"] = torch.from_numpy(row["audio"])
            logits = model(condition=condition, **kwargs)
            loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([row["target_index"]]))
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            optimizer.step()
            optimizer_updates += 1
            interval_loss_sum += float(loss.detach())
            interval_loss_count += 1
            if optimizer_updates % validation_interval == 0 or optimizer_updates == update_limit:
                validate(epoch)
                if stale_epochs >= patience:
                    break
        if optimizer_updates >= update_limit or stale_epochs >= patience:
            break
    if max_updates is not None and optimizer_updates < update_limit and stale_epochs < patience:
        raise RuntimeError("fixed update budget was not completed")

    model.eval()
    final_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    _atomic_torch_save(
        {
            "state_dict": final_state,
            "condition": condition,
            "seed": seed,
            "epoch": epoch,
            "optimizer_updates": optimizer_updates,
            "hidden_size": hidden_size,
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
        },
        Path(output_dir) / "final.pt",
    )
    if best_state is None or best_predictions is None:
        raise RuntimeError("training did not produce a selectable checkpoint")
    best_checkpoint = torch.load(
        Path(output_dir) / "best.pt", map_location="cpu", weights_only=True
    )
    if (
        best_checkpoint["condition"] != condition
        or best_checkpoint["epoch"] != best_epoch
        or best_checkpoint["optimizer_updates"] != best_step
    ):
        raise RuntimeError("saved best checkpoint metadata differs from selector state")
    model.load_state_dict(best_checkpoint["state_dict"], strict=True)
    replay_metrics, replay_predictions = _score_rows(model, selection_rows, condition)
    if replay_predictions != best_predictions or replay_metrics["nll"] != best_nll:
        raise RuntimeError("best checkpoint reload changed selection predictions")
    (Path(output_dir) / "history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (Path(output_dir) / "best-selection-predictions.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        for prediction in best_predictions:
            stream.write(json.dumps(prediction, sort_keys=True) + "\n")
    held_out_metrics = {}
    held_out_files = {}
    for split, filename in (
        ("calibration", "calibration-predictions.jsonl"),
        ("development_validation", "development-validation-predictions.jsonl"),
    ):
        held_out = [row for row in rows if row["split"] == split]
        if held_out:
            metrics, predictions = _score_rows(model, held_out, condition)
            held_out_metrics[split] = metrics
            path = Path(output_dir) / filename
            with path.open("w", encoding="utf-8", newline="\n") as stream:
                for prediction in predictions:
                    stream.write(json.dumps(prediction, sort_keys=True) + "\n")
            held_out_files[filename] = sha256_file(path)
    result = {
        "status": "PASS_AV_EVENT_READOUT_FIT",
        "condition": condition,
        "seed": seed,
        "fit_events": len(fit_rows),
        "selection_events": len(selection_rows),
        "fit_videos": len({row["video_id"] for row in fit_rows}),
        "selection_videos": len({row["video_id"] for row in selection_rows}),
        "best_epoch": best_epoch,
        "best_optimizer_updates": best_step,
        "best_selection_metrics": replay_metrics,
        "calibration_metrics": held_out_metrics.get("calibration"),
        "development_validation_metrics": held_out_metrics.get("development_validation"),
        "checkpoint_reload_verified": True,
        "epochs_completed": epoch,
        "seconds": time.perf_counter() - started,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "optimizer_updates": optimizer_updates,
        "max_updates": max_updates,
        "evaluation_interval": validation_interval,
        "train_ce_semantics": (
            "mean over optimizer updates since the previous validation; not a fixed-example gap"
        ),
        "stopped_by_patience": stale_epochs >= patience,
        "early_stopping": {"patience": patience, "min_delta": min_delta},
        "selection_rule": "minimum video-disjoint selection NLL with strict absolute min_delta",
    }
    result["artifacts_sha256"] = {
        name: sha256_file(Path(output_dir) / name)
        for name in ("best.pt", "final.pt", "history.json", "best-selection-predictions.jsonl")
    }
    result["artifacts_sha256"].update(held_out_files)
    (Path(output_dir) / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--condition", choices=("joint", "video_only"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--evaluation-interval", type=int)
    args = parser.parse_args()
    rows = load_feature_rows(args.features)
    result = fit_readout(
        rows,
        args.output_dir,
        condition=args.condition,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        min_delta=args.min_delta,
        hidden_size=args.hidden_size,
        max_updates=args.max_updates,
        evaluation_interval=args.evaluation_interval,
    )
    result["feature_manifest_sha256"] = sha256_file(args.features)
    result["source_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    result["runner_sha256"] = sha256_file(Path(__file__))
    result["feature_manifest"] = str(args.features.resolve())
    result["config"] = {
        "condition": args.condition,
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "min_delta": args.min_delta,
        "hidden_size": args.hidden_size,
        "max_updates": args.max_updates,
        "evaluation_interval": args.evaluation_interval,
        "device": "cpu",
        "optimizer": "AdamW",
        "selection_split": "video-disjoint selection",
        "evaluation_split": "not loaded by trainer",
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    _main()
