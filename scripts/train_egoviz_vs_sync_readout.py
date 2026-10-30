"""Fit small CPU readouts over frozen EgoViz video and sensor features."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

SEED = 17


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class JointTemporalScorer(nn.Module):
    """Project and compare ordered video frames to ordered sensor patches."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.video_projection = nn.Linear(768, hidden_size, bias=False)
        self.sensor_projection = nn.Linear(512, hidden_size, bias=False)

    def forward(self, video: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        # MOMENT channels are averaged, then adjacent sensor patches align to video frames.
        sensor = candidates.mean(dim=2).reshape(-1, 4, 8, 2, 512).mean(dim=3)
        video_embedding = nn.functional.normalize(self.video_projection(video), dim=-1)
        sensor_embedding = nn.functional.normalize(self.sensor_projection(sensor), dim=-1)
        return (video_embedding[:, None] * sensor_embedding).sum(dim=-1).mean(dim=-1)


class SensorOnlyScorer(nn.Module):
    """Candidate-independent utility score; no video/query representation is available."""

    def __init__(self):
        super().__init__()
        self.score = nn.Linear(512, 1)

    def forward(self, candidates: torch.Tensor) -> torch.Tensor:
        sensor = candidates.mean(dim=(2, 3))
        return self.score(sensor).squeeze(-1)


def metrics(logits: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    one_hot = np.eye(logits.shape[1])[targets]
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    ece = 0.0
    for lower in np.linspace(0, 1, 11)[:-1]:
        upper = lower + 0.1
        upper_test = confidence < upper if upper < 1 else confidence <= upper
        selected = (confidence >= lower) & upper_test
        if selected.any():
            ece += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return {
        "accuracy": float(correct.mean()),
        "nll": float(
            -np.log(np.maximum(probabilities[np.arange(len(targets)), targets], 1e-15)).mean()
        ),
        "brier": float(((probabilities - one_hot) ** 2).sum(axis=1).mean()),
        "ece": float(ece),
        "mean_confidence": float(confidence.mean()),
        "count": int(len(targets)),
    }


def uniform_metrics(count: int, classes: int = 4) -> dict[str, float]:
    """Expected metrics for tied uniform choice; avoid arbitrary argmax tie-breaking."""
    return {
        "accuracy": 1.0 / classes,
        "nll": float(np.log(classes)),
        "brier": 1.0 - 1.0 / classes,
        "ece": 0.0,
        "mean_confidence": 1.0 / classes,
        "count": count,
    }


def _load_cache(cache_path: Path, expected_name: str) -> dict[str, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _load_samples(corpus_dir: Path, video_cache: Path, sensor_cache: Path):
    rows = [
        json.loads(line)
        for split in ("train", "development")
        for line in (corpus_dir / f"{split}-samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    video = _load_cache(video_cache, "video-features.npz")
    sensor = _load_cache(sensor_cache, "sensor-features.npz")
    if video["features"].shape != (len(video["sample_id"]), 8, 768):
        raise ValueError("unexpected cached VideoPrism feature shape")
    if sensor["features"].shape != (len(sensor["window_index"]), 6, 16, 512):
        raise ValueError("unexpected cached MOMENT feature shape")
    video_by_id = {
        str(key): value
        for key, value in zip(video["sample_id"], video["features"], strict=True)
    }
    if set(video_by_id) != {row["sample_id"] for row in rows}:
        raise ValueError("video cache sample IDs differ from frozen corpus")
    return rows, video_by_id, sensor["features"]


def _tensor_data(rows, video_by_id, sensor_features, split):
    selected = [row for row in rows if row["split"] == split]
    videos = np.stack([video_by_id[row["sample_id"]] for row in selected])
    candidates = np.stack(
        [sensor_features[row["candidate_window_indices"]] for row in selected]
    )
    labels = np.asarray([row["target_option_index"] for row in selected], dtype=np.int64)
    groups = np.asarray([row["group_id"] for row in selected])
    return (
        torch.from_numpy(videos).float(),
        torch.from_numpy(candidates).float(),
        torch.from_numpy(labels),
        groups,
        [row["sample_id"] for row in selected],
        np.asarray([row["query_index"] for row in selected], dtype=np.int64),
    )


def _fit(model, train_data, validation_data, *, joint: bool, max_epochs: int = 100):
    video_train, sensor_train, labels_train = train_data[:3]
    video_val, sensor_val, labels_val = validation_data[:3]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    best_state = None
    best_nll = float("inf")
    best_epoch = 0
    patience = 12
    stale = 0
    history = []
    started = time.perf_counter()
    batch_size = 32
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(labels_train))
        train_loss_sum = 0.0
        for indices in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            if joint:
                logits = model(video_train[indices], sensor_train[indices])
            else:
                logits = model(sensor_train[indices])
            loss = nn.functional.cross_entropy(logits, labels_train[indices])
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(indices)
        model.eval()
        with torch.inference_mode():
            val_logits = (
                model(video_val, sensor_val) if joint else model(sensor_val)
            ).cpu().numpy()
        val_metrics = metrics(val_logits, labels_val.numpy())
        history.append(
            {
                "epoch": epoch,
                "train_ce": train_loss_sum / len(labels_train),
                "validation": val_metrics,
            }
        )
        if val_metrics["nll"] < best_nll - 1e-6:
            best_nll = val_metrics["nll"]
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        train_logits = (model(video_train, sensor_train) if joint else model(sensor_train)).numpy()
        val_logits = (model(video_val, sensor_val) if joint else model(sensor_val)).numpy()
    return {
        "best_epoch": best_epoch,
        "stopped_epoch": epoch,
        "training_seconds": time.perf_counter() - started,
        "train_logits": train_logits,
        "validation_logits": val_logits,
        "history": history,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_state": best_state,
    }


def _group_bootstrap_delta(
    logits_a: np.ndarray,
    logits_b: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int = SEED,
    repetitions: int = 2000,
) -> dict[str, Any]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("group bootstrap requires at least two groups")
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repetitions):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in sampled_groups])
        accuracy_a = (logits_a[indices].argmax(axis=1) == targets[indices]).mean()
        accuracy_b = (logits_b[indices].argmax(axis=1) == targets[indices]).mean()
        differences.append(float(accuracy_a - accuracy_b))
    return {
        "method": "paired group bootstrap; resample development sessions with replacement",
        "seed": seed,
        "repetitions": repetitions,
        "group_count": len(unique_groups),
        "accuracy_delta_a_minus_b": float(
            (logits_a.argmax(axis=1) == targets).mean()
            - (logits_b.argmax(axis=1) == targets).mean()
        ),
        "percentile_95_ci": [
            float(np.quantile(differences, 0.025)),
            float(np.quantile(differences, 0.975)),
        ],
    }


def run(corpus_dir: Path, video_cache: Path, sensor_cache: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rows, video_by_id, sensor_features = _load_samples(corpus_dir, video_cache, sensor_cache)
    train_data = _tensor_data(rows, video_by_id, sensor_features, "train")
    dev_data = _tensor_data(rows, video_by_id, sensor_features, "development")
    y_train = train_data[2].numpy()
    y_dev = dev_data[2].numpy()

    joint = _fit(JointTemporalScorer(), train_data, dev_data, joint=True)
    sensor_only = _fit(SensorOnlyScorer(), train_data, dev_data, joint=False)
    joint_train = metrics(joint["train_logits"], y_train)
    sensor_train = metrics(sensor_only["train_logits"], y_train)
    joint_dev = metrics(joint["validation_logits"], y_dev)
    sensor_dev = metrics(sensor_only["validation_logits"], y_dev)
    groups = dev_data[3]
    video_val, sensor_val = dev_data[0], dev_data[1]
    query_indices = dev_data[5]
    mismatched_video = video_val.clone()
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        ordered = indices[np.argsort(query_indices[indices])]
        shift = max(1, len(ordered) // 2)
        mismatched_video[ordered] = video_val[np.roll(ordered, shift)]
    joint_model = JointTemporalScorer()
    joint_model.load_state_dict(joint["best_state"])
    joint_model.eval()
    with torch.inference_mode():
        mismatch_logits = joint_model(mismatched_video, sensor_val).numpy()
        permutation = torch.tensor([2, 0, 3, 1])
        inverse_permutation = torch.argsort(permutation)
        permuted_logits = joint_model(video_val, sensor_val[:, permutation]).numpy()
    np.testing.assert_allclose(
        permuted_logits,
        joint["validation_logits"][:, permutation.numpy()],
        rtol=1e-5,
        atol=1e-6,
    )
    permuted_metrics = metrics(permuted_logits, inverse_permutation[y_dev].numpy())
    mismatch_metrics = metrics(mismatch_logits, y_dev)
    output.mkdir(parents=True, exist_ok=False)
    joint_state_path = output / "joint-temporal-state.pt"
    sensor_state_path = output / "sensor-only-state.pt"
    torch.save(joint["best_state"], joint_state_path)
    torch.save(sensor_only["best_state"], sensor_state_path)
    predictions = []
    probabilities = torch.softmax(torch.from_numpy(joint["validation_logits"]), dim=1).numpy()
    mismatch_probabilities = torch.softmax(torch.from_numpy(mismatch_logits), dim=1).numpy()
    for index, sample_id in enumerate(dev_data[4]):
        predictions.append(
            {
                "sample_id": sample_id,
                "group_id": str(groups[index]),
                "target": int(y_dev[index]),
                "joint_prediction": int(joint["validation_logits"][index].argmax()),
                "joint_probabilities": probabilities[index].tolist(),
                "sensor_only_prediction": int(sensor_only["validation_logits"][index].argmax()),
                "mismatched_video_prediction": int(mismatch_logits[index].argmax()),
                "mismatched_video_probabilities": mismatch_probabilities[index].tolist(),
            }
        )
    (output / "development-predictions.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in predictions),
        encoding="utf-8",
    )
    report = {
        "status": "EXPLORATORY_DEVELOPMENT_READOUTS_NOT_PRODUCT_QUALITY",
        "seed": SEED,
        "device": "cpu",
        "runner_sha256": sha256_file(Path(__file__)),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "training_config": {
            "optimizer": "AdamW",
            "learning_rate": 0.003,
            "weight_decay": 0.01,
            "batch_size": 32,
            "max_epochs": 100,
            "early_stopping": "development NLL; patience 12; min_delta 1e-6",
            "joint_model": "linear 768->64 and 512->64; normalized temporal dot product",
            "sensor_only_model": "linear 512->1 on mean-pooled MOMENT channels and patches",
            "loss": "four-choice cross entropy",
        },
        "input_sha256": {
            "train_samples": sha256_file(corpus_dir / "train-samples.jsonl"),
            "development_samples": sha256_file(corpus_dir / "development-samples.jsonl"),
            "video_features": sha256_file(video_cache),
            "sensor_features": sha256_file(sensor_cache),
        },
        "train_groups": int(len({row["group_id"] for row in rows if row["split"] == "train"})),
        "development_groups": int(len(np.unique(groups))),
        "training_examples": int(len(y_train)),
        "development_examples": int(len(y_dev)),
        "video_only_uniform": uniform_metrics(len(y_dev)),
        "sensor_only": {
            "parameter_count": sensor_only["parameter_count"],
            "best_epoch": sensor_only["best_epoch"],
            "stopped_epoch": sensor_only["stopped_epoch"],
            "train_metrics": sensor_train,
            "development_metrics": sensor_dev,
            "training_seconds": sensor_only["training_seconds"],
            "history": sensor_only["history"],
            "state_file": sensor_state_path.name,
            "state_sha256": sha256_file(sensor_state_path),
        },
        "joint_temporal": {
            "parameter_count": joint["parameter_count"],
            "best_epoch": joint["best_epoch"],
            "stopped_epoch": joint["stopped_epoch"],
            "train_metrics": joint_train,
            "development_metrics": joint_dev,
            "training_seconds": joint["training_seconds"],
            "history": joint["history"],
            "state_file": joint_state_path.name,
            "state_sha256": sha256_file(joint_state_path),
        },
        "joint_vs_sensor_only_accuracy_group_bootstrap": _group_bootstrap_delta(
            joint["validation_logits"], sensor_only["validation_logits"], y_dev, groups
        ),
        "joint_with_mismatched_video": mismatch_metrics,
        "joint_vs_mismatched_video_accuracy_group_bootstrap": _group_bootstrap_delta(
            joint["validation_logits"], mismatch_logits, y_dev, groups
        ),
        "option_permutation_equivariance": {
            "permutation": permutation.tolist(),
            "metrics_after_remapping_targets": permuted_metrics,
            "logits_permute_equivariance": "PASS",
        },
        "limitations": [
            f"development has {len(np.unique(groups))} publisher session groups; "
            "participant independence is unknown",
            "timestamp correspondence is not a general multimodal decision benchmark",
            "one seed and validation-based early stopping are exploratory only",
        ],
        "sealed_audit_accessed": False,
    }
    (output / "readout-results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--video-cache", type=Path, required=True)
    parser.add_argument("--sensor-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.corpus, args.video_cache, args.sensor_cache, args.output), indent=2))


if __name__ == "__main__":
    main()
