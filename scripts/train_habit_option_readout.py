"""Train fixed-budget option readouts over pinned frozen HABIT features.

This is an exploratory transition-lookahead task. It reads train/validation
feature shards only and never loads the reserved split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

MODES = ("options_only", "video_mean", "state_mean", "video_state_mean", "video_state_lstm")
SEED = 17
HIDDEN = 64
EPOCHS = 25
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
EXPECTED_SAMPLES = {"train": 936, "validation": 144}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classification_metrics(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if probabilities.ndim != 2 or len(targets) != len(probabilities):
        raise ValueError("probability/target shape mismatch")
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-5
    ):
        raise ValueError("invalid probability rows")
    rows = np.arange(len(targets))
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    one_hot = np.eye(probabilities.shape[1])[targets]
    ece = 0.0
    for lower in np.linspace(0.0, 1.0, 16)[:-1]:
        upper = lower + 1 / 15
        selected = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= 1)
        if selected.any():
            ece += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return {
        "count": int(len(targets)),
        "accuracy": float(correct.mean()),
        "nll": float(-np.log(np.clip(probabilities[rows, targets], 1e-12, 1)).mean()),
        "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ece_15": float(ece),
        "mean_confidence": float(confidence.mean()),
    }


def cluster_bootstrap_delta(
    left_correct: np.ndarray,
    right_correct: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int = SEED,
    repetitions: int = 10_000,
) -> dict[str, Any]:
    labels = np.unique(groups)
    if len(labels) < 2:
        raise ValueError("cluster bootstrap requires at least two groups")
    per_group = {
        label: float(left_correct[groups == label].mean() - right_correct[groups == label].mean())
        for label in labels
    }
    values = np.asarray([per_group[label] for label in labels])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(labels), size=(repetitions, len(labels)))
    means = values[draws].mean(axis=1)
    return {
        "unit": "task_group",
        "groups": [str(label) for label in labels],
        "per_group_delta": {str(key): value for key, value in per_group.items()},
        "mean_delta": float(values.mean()),
        "bootstrap_repetitions": repetitions,
        "percentile_95": [float(x) for x in np.quantile(means, [0.025, 0.975])],
        "warning": "Only three validation task groups; interval is descriptive and unstable.",
    }


class OptionReadout(nn.Module):
    def __init__(self, mode: str, max_option_id: int, hidden: int = HIDDEN) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unsupported readout mode: {mode}")
        self.mode = mode
        self.hidden = hidden
        temporal_dim = {
            "video_mean": 768,
            "state_mean": 42,
            "video_state_mean": 810,
            "video_state_lstm": 810,
        }.get(mode)
        self.temporal_proj = nn.Linear(temporal_dim, hidden) if temporal_dim else None
        self.temporal = (
            nn.LSTM(hidden, hidden, batch_first=True) if mode == "video_state_lstm" else None
        )
        self.global_query = nn.Parameter(torch.zeros(1, hidden)) if mode == "options_only" else None
        self.option_proj = nn.Sequential(nn.Linear(384, hidden), nn.LayerNorm(hidden))
        self.action_bias = nn.Embedding(max_option_id + 1, 1)
        nn.init.zeros_(self.action_bias.weight)

    def forward(
        self,
        video: torch.Tensor,
        state: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_ids: torch.Tensor,
        option_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "options_only":
            query = self.global_query.expand(video.shape[0], -1)
        else:
            if self.mode == "video_mean":
                temporal_input = video
            elif self.mode == "state_mean":
                temporal_input = state
            else:
                temporal_input = torch.cat((video, state), dim=-1)
            encoded = torch.tanh(self.temporal_proj(temporal_input))
            if self.temporal is None:
                query = encoded.mean(dim=1)
            else:
                _, (hidden, _) = self.temporal(encoded)
                query = hidden[-1]
        options = self.option_proj(candidate_features)
        logits = torch.einsum("bd,bod->bo", query, options) / math.sqrt(self.hidden)
        logits = logits + self.action_bias(candidate_ids.clamp_min(0)).squeeze(-1)
        return logits.masked_fill(~option_mask, -1e9)


def load_splits(cache_dir: Path, expected_config_sha256: str) -> dict[str, dict[str, Any]]:
    progress_path = cache_dir / "progress.jsonl"
    progress_hashes: dict[tuple[str, int], str] = {}
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("event") == "episode_complete":
            progress_hashes[(event["split"], int(event["episode_id"]))] = event["shard_sha256"]
    result = {}
    for split in ("train", "validation"):
        arrays: dict[str, list[np.ndarray]] = {
            key: []
            for key in (
                "video_features",
                "state_features",
                "candidate_features",
                "candidate_ids",
                "option_mask",
                "target_positions",
                "target_ids",
                "sample_ids",
                "group_ids",
            )
        }
        shard_paths = sorted((cache_dir / "shards" / split).glob("episode_*.npz"))
        if not shard_paths:
            raise FileNotFoundError(f"no {split} feature shards")
        for shard_path in shard_paths:
            episode_id = int(shard_path.stem.split("_")[1])
            expected_hash = progress_hashes.get((split, episode_id))
            if expected_hash is None or sha256_file(shard_path) != expected_hash:
                raise ValueError(f"feature shard SHA mismatch: {shard_path}")
            with np.load(shard_path, allow_pickle=False) as shard:
                if shard["config_sha256"].item() != expected_config_sha256:
                    raise ValueError(f"feature config mismatch: {shard_path}")
                if (
                    not np.isfinite(shard["video_features"]).all()
                    or not np.isfinite(shard["state_features"]).all()
                ):
                    raise FloatingPointError(f"non-finite feature: {shard_path}")
                for key in arrays:
                    arrays[key].append(shard[key])
        option_width = max(value.shape[1] for value in arrays["candidate_ids"])
        for index, value in enumerate(arrays["candidate_features"]):
            difference = option_width - value.shape[1]
            if difference:
                arrays["candidate_features"][index] = np.pad(
                    value, ((0, 0), (0, difference), (0, 0))
                )
        for key, fill in (("candidate_ids", -1), ("option_mask", False)):
            for index, value in enumerate(arrays[key]):
                difference = option_width - value.shape[1]
                if difference:
                    arrays[key][index] = np.pad(
                        value, ((0, 0), (0, difference)), constant_values=fill
                    )
        combined = {key: np.concatenate(values, axis=0) for key, values in arrays.items()}
        if len(combined["sample_ids"]) != EXPECTED_SAMPLES[split]:
            raise ValueError(f"unexpected {split} row count")
        if len(set(combined["sample_ids"].tolist())) != EXPECTED_SAMPLES[split]:
            raise ValueError(f"duplicate {split} sample IDs")
        if (combined["target_positions"] >= combined["option_mask"].sum(axis=1)).any():
            raise ValueError(f"target outside option set in {split}")
        result[split] = combined
    train_ids = set(result["train"]["sample_ids"].tolist())
    valid_ids = set(result["validation"]["sample_ids"].tolist())
    if train_ids & valid_ids:
        raise ValueError("train/validation sample overlap")
    return result


def tensors(rows: dict[str, Any]) -> TensorDataset:
    return TensorDataset(
        torch.as_tensor(rows["video_features"], dtype=torch.float32),
        torch.as_tensor(rows["state_features"], dtype=torch.float32),
        torch.as_tensor(rows["candidate_features"], dtype=torch.float32),
        torch.as_tensor(rows["candidate_ids"], dtype=torch.long),
        torch.as_tensor(rows["option_mask"], dtype=torch.bool),
        torch.as_tensor(rows["target_positions"], dtype=torch.long),
    )


@torch.no_grad()
def evaluate(
    model: OptionReadout,
    data: TensorDataset,
    rows: dict[str, Any],
    *,
    batch_size: int = BATCH_SIZE,
    intervention: str = "none",
) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    outputs = []
    start = 0
    for video, state, candidates, candidate_ids, masks, _ in loader:
        if intervention in ("zero_video", "shuffle_video"):
            if intervention == "zero_video":
                video = torch.zeros_like(video)
            else:
                perm = torch.as_tensor(
                    np.random.default_rng(SEED + 100).permutation(len(rows["sample_ids"]))[
                        start : start + len(video)
                    ],
                    dtype=torch.long,
                )
                video = torch.as_tensor(rows["video_features"][perm.numpy()])
        if intervention in ("zero_state", "shuffle_state"):
            if intervention == "zero_state":
                state = torch.zeros_like(state)
            else:
                perm = torch.as_tensor(
                    np.random.default_rng(SEED + 200).permutation(len(rows["sample_ids"]))[
                        start : start + len(state)
                    ],
                    dtype=torch.long,
                )
                state = torch.as_tensor(rows["state_features"][perm.numpy()])
        logits = model(video, state, candidates, candidate_ids, masks)
        outputs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        start += len(video)
    probabilities = np.concatenate(outputs)
    targets = rows["target_positions"].astype(np.int64)
    return classification_metrics(probabilities, targets), probabilities


def _train_one(
    mode: str,
    train_rows: dict[str, Any],
    valid_rows: dict[str, Any],
    output_dir: Path,
    progress_path: Path,
) -> dict[str, Any]:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    model = OptionReadout(mode, int(train_rows["candidate_ids"].max())).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_data, valid_data = tensors(train_rows), tensors(valid_rows)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    history = []
    best_nll = float("inf")
    best_epoch = 0
    best_state = None
    started = time.perf_counter()
    best_model_path = output_dir / f"{mode}-best.pt"
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for video, state, candidates, candidate_ids, masks, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(video, state, candidates, candidate_ids, masks)
            loss = nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss: {mode} epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        train_metrics, train_probs = evaluate(model, train_data, train_rows)
        validation_metrics, validation_probs = evaluate(model, valid_data, valid_rows)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "train_metrics_resubstitution": train_metrics,
                "validation_metrics": validation_metrics,
            }
        )
        with progress_path.open("a", encoding="utf-8", buffering=1) as progress:
            progress.write(
                json.dumps(
                    {
                        "mode": mode,
                        "epoch": epoch,
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "train_loss": float(np.mean(losses)),
                        "train_metrics_resubstitution": train_metrics,
                        "validation_metrics": validation_metrics,
                    }
                )
                + "\n"
            )
        if validation_metrics["nll"] < best_nll:
            best_nll = validation_metrics["nll"]
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            temporary_checkpoint = best_model_path.with_suffix(".tmp.pt")
            torch.save(best_state, temporary_checkpoint)
            temporary_checkpoint.replace(best_model_path)
            np.savez_compressed(
                output_dir / f"{mode}-best-predictions.npz",
                sample_ids=valid_rows["sample_ids"],
                target_positions=valid_rows["target_positions"],
                target_ids=valid_rows["target_ids"],
                group_ids=valid_rows["group_ids"],
                probabilities=validation_probs,
            )
    if best_state is None:
        raise RuntimeError("no best-validation checkpoint selected")
    model.load_state_dict(best_state)
    best_metrics, _ = evaluate(model, valid_data, valid_rows)
    return {
        "mode": mode,
        "selection": "minimum validation NLL over 25 epochs",
        "selected_epoch": best_epoch,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_validation_metrics": best_metrics,
        "train_metrics_at_selected_epoch_resubstitution": history[best_epoch - 1][
            "train_metrics_resubstitution"
        ],
        "history": history,
        "checkpoint": best_model_path.name,
        "checkpoint_sha256": sha256_file(best_model_path),
        "validation_predictions_sha256": sha256_file(output_dir / f"{mode}-best-predictions.npz"),
        "wall_seconds": round(time.perf_counter() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cache_dir = args.feature_cache.resolve(strict=True)
    receipt_path = cache_dir / "receipt.json"
    receipt_sha256 = sha256_file(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "complete_habit_frozen_videoprism_feature_cache"
        or receipt.get("reserved_loaded")
        or receipt.get("train_samples") != EXPECTED_SAMPLES["train"]
        or receipt.get("validation_samples") != EXPECTED_SAMPLES["validation"]
    ):
        raise ValueError("feature cache receipt is not complete or has wrong split sizes")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    run_config = {
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "hidden_size": HIDDEN,
        "modes": list(MODES),
        "device": "cpu",
        "selection": "minimum validation NLL over fixed 25-epoch budget",
        "feature_receipt_sha256": receipt_sha256,
        "train_samples": EXPECTED_SAMPLES["train"],
        "validation_samples": EXPECTED_SAMPLES["validation"],
        "reserved_loaded": False,
        "training_metrics_note": "Resubstitution metrics; not fixed-example generalization gap.",
    }
    config_path = output_dir / "run-config.json"
    config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    config_sha256 = sha256_file(config_path)
    splits = load_splits(cache_dir, receipt["config_sha256"])
    results = []
    progress_path = output_dir / "training-history.jsonl"
    for mode in MODES:
        result = _train_one(mode, splits["train"], splits["validation"], output_dir, progress_path)
        result["validation_group_metrics"] = {}
        model = OptionReadout(mode, int(splits["train"]["candidate_ids"].max())).cpu()
        model.load_state_dict(
            torch.load(output_dir / result["checkpoint"], map_location="cpu", weights_only=True)
        )
        valid_data = tensors(splits["validation"])
        _, probabilities = evaluate(model, valid_data, splits["validation"])
        predicted = probabilities.argmax(axis=1)
        target = splits["validation"]["target_positions"]
        for group in np.unique(splits["validation"]["group_ids"]):
            selected = splits["validation"]["group_ids"] == group
            result["validation_group_metrics"][str(group)] = classification_metrics(
                probabilities[selected], target[selected]
            )
        result["validation_interventions"] = {}
        for intervention in ("zero_video", "shuffle_video", "zero_state", "shuffle_state"):
            result["validation_interventions"][intervention] = evaluate(
                model,
                valid_data,
                splits["validation"],
                intervention=intervention,
            )[0]
        result["validation_predictions"] = probabilities.tolist()
        result["validation_correct"] = (predicted == target).tolist()
        result["validation_sample_ids"] = splits["validation"]["sample_ids"].tolist()
        result["validation_group_ids"] = splits["validation"]["group_ids"].tolist()
        results.append(result)
    fused = next(row for row in results if row["mode"] == "video_state_lstm")
    baseline = next(row for row in results if row["mode"] == "options_only")
    fused_correct = np.asarray(fused["validation_correct"], dtype=bool)
    baseline_correct = np.asarray(baseline["validation_correct"], dtype=bool)
    groups = np.asarray(fused["validation_group_ids"])
    report = {
        "status": "complete_exploratory_habit_option_readout",
        "run_config_sha256": config_sha256,
        "feature_cache_receipt_sha256": receipt_sha256,
        "selection_warning": (
            "All metrics use the small development validation split and influence "
            "checkpoint selection."
        ),
        "run_order": list(MODES),
        "results": results,
        "fused_lstm_vs_options_only_accuracy_cluster_bootstrap": cluster_bootstrap_delta(
            fused_correct, baseline_correct, groups
        ),
        "input_control_warning": (
            "Zero/shuffle controls are diagnostics; groups and task are fixed. "
            "The validation set has three task groups."
        ),
        "package_versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "platform": platform.platform(),
        "device": "cpu",
    }
    (output_dir / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "config_sha256": config_sha256}), flush=True)


if __name__ == "__main__":
    main()
