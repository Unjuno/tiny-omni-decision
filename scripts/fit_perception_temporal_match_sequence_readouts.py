"""Fit matched audio-only, pooled AV, and sequence-aware AV readouts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SEED = 17
TOTAL_UPDATES = 2_048
EVALUATION_INTERVAL = 32
LEARNING_RATE = 0.002
WEIGHT_DECAY = 1e-3
HIDDEN_DIMENSION = 32


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def categorical_metrics(scores: np.ndarray, targets: np.ndarray, bins: int = 10) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("scores must be a nonempty [examples, options] matrix")
    if labels.shape != (values.shape[0],):
        raise ValueError("targets must have one entry per score row")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    if np.any(labels < 0) or np.any(labels >= values.shape[1]):
        raise ValueError("target index is outside the option range")
    if bins < 1:
        raise ValueError("bins must be positive")

    shifted = values - values.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    one_hot = np.eye(values.shape[1], dtype=np.float64)[labels]
    ece = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        in_bin = (confidence >= low) & (
            confidence <= high if index == bins - 1 else confidence < high
        )
        if np.any(in_bin):
            ece += float(in_bin.mean()) * abs(
                float(correct[in_bin].mean()) - float(confidence[in_bin].mean())
            )
    return {
        "count": int(labels.size),
        "accuracy": float(correct.mean()),
        "nll": float(
            -np.log(np.maximum(probabilities[np.arange(labels.size), labels], 1e-15)).mean()
        ),
        "brier": float(((probabilities - one_hot) ** 2).sum(axis=1).mean()),
        "ece_10": float(ece),
        "mean_confidence": float(confidence.mean()),
        "probabilities": probabilities,
    }


class TemporalCandidateScorer(nn.Module):
    """Score each audio option against a video sequence using a conditioned LSTM."""

    def __init__(
        self,
        video_dim: int = 768,
        audio_dim: int = 384,
        hidden_dim: int = HIDDEN_DIMENSION,
        pool_video: bool = False,
    ) -> None:
        super().__init__()
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.pool_video = pool_video
        self.video_projection = nn.Linear(video_dim, hidden_dim)
        self.audio_projection = nn.Linear(audio_dim, hidden_dim)
        self.initial_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.initial_cell = nn.Linear(hidden_dim, hidden_dim)
        self.temporal = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        if video.ndim != 3 or video.shape[-1] != self.video_dim or video.shape[1] < 1:
            raise ValueError("video must have shape [batch, time, video_dim]")
        if audio.ndim != 3 or audio.shape[-1] != self.audio_dim or audio.shape[1] < 2:
            raise ValueError("audio must have shape [batch, options, audio_dim]")
        if video.shape[0] != audio.shape[0]:
            raise ValueError("video and audio batch sizes must match")
        if not torch.isfinite(video).all() or not torch.isfinite(audio).all():
            raise ValueError("features must be finite")
        if self.pool_video:
            video = video.mean(dim=1, keepdim=True)

        batch, time_steps, _ = video.shape
        option_count = audio.shape[1]
        video_projected = torch.tanh(self.video_projection(video))
        audio_projected = torch.tanh(self.audio_projection(audio))
        video_repeated = (
            video_projected[:, None]
            .expand(batch, option_count, time_steps, self.hidden_dim)
            .reshape(batch * option_count, time_steps, self.hidden_dim)
        )
        audio_flat = audio_projected.reshape(batch * option_count, self.hidden_dim)
        hidden = torch.tanh(self.initial_hidden(audio_flat)).unsqueeze(0)
        cell = torch.tanh(self.initial_cell(audio_flat)).unsqueeze(0)
        sequence, _ = self.temporal(video_repeated, (hidden, cell))
        attention = torch.softmax(
            (sequence * audio_flat[:, None]).sum(dim=-1) / math.sqrt(self.hidden_dim), dim=1
        )
        context = (sequence * attention[..., None]).sum(dim=1)
        combined = torch.cat((context, audio_flat, context * audio_flat), dim=-1)
        return self.scorer(combined).reshape(batch, option_count)


class AudioCandidateScorer(nn.Module):
    def __init__(self, audio_dim: int = 384, hidden_dim: int = HIDDEN_DIMENSION) -> None:
        super().__init__()
        self.audio_dim = audio_dim
        self.scorer = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim != 3 or audio.shape[-1] != self.audio_dim or audio.shape[1] < 2:
            raise ValueError("audio must have shape [batch, options, audio_dim]")
        if not torch.isfinite(audio).all():
            raise ValueError("features must be finite")
        return self.scorer(audio).squeeze(-1)


def _load_feature(path: Path, expected_hash: str, expected_shape: tuple[int, ...]) -> np.ndarray:
    array = np.load(path.resolve(strict=True), allow_pickle=False)
    values = np.asarray(array, dtype=np.float32)
    if values.shape != expected_shape or not np.isfinite(values).all():
        raise ValueError(f"invalid feature shape or values: {path}")
    if hashlib.sha256(values.tobytes()).hexdigest() != expected_hash:
        raise ValueError(f"feature hash mismatch: {path}")
    return values


def load_dataset(manifest_path: Path, video_result_path: Path, audio_result_path: Path):
    manifest_path = manifest_path.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    video_result = json.loads(video_result_path.resolve(strict=True).read_text(encoding="utf-8"))
    audio_result = json.loads(audio_result_path.resolve(strict=True).read_text(encoding="utf-8"))
    manifest_hash = sha256_file(manifest_path)
    if video_result.get("manifest_sha256") != manifest_hash:
        raise ValueError("VideoPrism feature manifest hash mismatch")
    if audio_result.get("manifest_sha256") != manifest_hash:
        raise ValueError("EfficientAT feature manifest hash mismatch")
    if video_result.get("feature_layout") != "per_frame_mean":
        raise ValueError("temporal readout requires per_frame_mean video features")
    video_features = {}
    for row in video_result.get("records", []):
        sample_id = row["sample_id"]
        if sample_id in video_features:
            raise ValueError(f"duplicate video feature row: {sample_id}")
        video_features[sample_id] = _load_feature(
            Path(row["feature_path"]), row["feature_sha256"], (8, 768)
        )
    audio_features = {}
    for row in audio_result.get("records", []):
        key = (row["sample_id"], int(row["event_id"]))
        if key in audio_features:
            raise ValueError(f"duplicate audio feature row: {key}")
        audio_features[key] = _load_feature(
            Path(row["feature_path"]), row["feature_sha256"], (384,)
        )

    result = {}
    split_ids: dict[str, set[str]] = {}
    for split in ("train", "validation"):
        rows = manifest.get(split)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"manifest has no {split} samples")
        split_ids[split] = {str(row["video_id"]) for row in rows}
        if len(split_ids[split]) != len(rows):
            raise ValueError(f"duplicate video groups in {split}")
        samples = []
        for row in rows:
            sample_id = row["sample_id"]
            if sample_id not in video_features:
                raise ValueError(f"missing video feature: {sample_id}")
            options = row["options"]
            if len(options) != 2:
                raise ValueError(f"expected two audio options: {sample_id}")
            vectors = []
            for option in options:
                key = (sample_id, int(option["event_id"]))
                if key not in audio_features:
                    raise ValueError(f"missing audio feature: {key}")
                vectors.append(audio_features[key])
            target = int(row["target_index"])
            if target not in (0, 1):
                raise ValueError(f"invalid binary target: {sample_id}")
            samples.append(
                {
                    "sample_id": sample_id,
                    "video_id": str(row["video_id"]),
                    "target": target,
                    "video": video_features[sample_id],
                    "audio": np.stack(vectors),
                }
            )
        result[split] = samples
    if split_ids["train"] & split_ids["validation"]:
        raise ValueError("train/validation video-group overlap")
    if set(video_features) != {
        row["sample_id"] for split in ("train", "validation") for row in manifest[split]
    }:
        raise ValueError("unexpected or missing VideoPrism feature rows")
    return manifest, result


def _arrays(samples: list[dict[str, Any]], video_mean, video_std, audio_mean, audio_std):
    video = np.stack([row["video"] for row in samples])
    audio = np.stack([row["audio"] for row in samples])
    video = (video - video_mean) / video_std
    audio = (audio - audio_mean) / audio_std
    targets = np.asarray([row["target"] for row in samples], dtype=np.int64)
    return (
        torch.from_numpy(video.astype(np.float32, copy=False)),
        torch.from_numpy(audio.astype(np.float32, copy=False)),
        torch.from_numpy(targets),
    )


def _predict(model, video: torch.Tensor | None, audio: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        scores = model(video, audio) if video is not None else model(audio)
    output = scores.detach().cpu().numpy()
    if not np.isfinite(output).all():
        raise RuntimeError("readout produced non-finite scores")
    return output


def _evaluate(model, video: torch.Tensor | None, audio: torch.Tensor, targets: torch.Tensor):
    scores = _predict(model, video, audio)
    return scores, categorical_metrics(scores, targets.numpy())


def _fit_arm(name, model, train, validation, train_rows, validation_rows, output, history_path):
    train_video, train_audio, train_targets = train
    valid_video, valid_audio, valid_targets = validation
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best_state = copy.deepcopy(model.state_dict())
    best_step = 0
    best_scores, best_metrics = _evaluate(model, valid_video, valid_audio, valid_targets)
    best_nll = best_metrics["nll"]
    arm_dir = output / name
    arm_dir.mkdir()
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as history:

        def record(step: int) -> None:
            train_scores, train_metrics = _evaluate(model, train_video, train_audio, train_targets)
            valid_scores, valid_metrics = _evaluate(model, valid_video, valid_audio, valid_targets)
            history.write(
                json.dumps(
                    {
                        "arm": name,
                        "step": step,
                        "train": {k: v for k, v in train_metrics.items() if k != "probabilities"},
                        "validation": {
                            k: v for k, v in valid_metrics.items() if k != "probabilities"
                        },
                    }
                )
                + "\n"
            )
            history.flush()
            return train_scores, train_metrics, valid_scores, valid_metrics

        _, _, _, initial_metrics = record(0)
        best_metrics = initial_metrics
        best_nll = initial_metrics["nll"]
        for step in range(1, TOTAL_UPDATES + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            scores = (
                model(train_video, train_audio) if train_video is not None else model(train_audio)
            )
            loss = F.cross_entropy(scores, train_targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss in {name} at step {step}")
            loss.backward()
            optimizer.step()
            if step % EVALUATION_INTERVAL == 0 or step == TOTAL_UPDATES:
                _, _, _, current_metrics = record(step)
                if current_metrics["nll"] < best_nll - 1e-12:
                    best_nll = current_metrics["nll"]
                    best_step = step
                    best_state = copy.deepcopy(model.state_dict())
                    best_metrics = current_metrics
        torch.save(
            {"state_dict": best_state, "arm": name, "selected_step": best_step, "seed": SEED},
            arm_dir / "best-checkpoint.pt",
        )
        torch.save(
            {"state_dict": model.state_dict(), "arm": name, "step": TOTAL_UPDATES, "seed": SEED},
            arm_dir / "final-checkpoint.pt",
        )
        model.load_state_dict(best_state, strict=True)
        reloaded_scores, reloaded_metrics = _evaluate(
            model, valid_video, valid_audio, valid_targets
        )
        if abs(reloaded_metrics["nll"] - best_nll) > 1e-10:
            raise RuntimeError(f"best checkpoint reload mismatch for {name}")

    return {
        "model": model,
        "selected_step": best_step,
        "metrics": {k: v for k, v in reloaded_metrics.items() if k != "probabilities"},
        "validation_scores": reloaded_scores,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
    }


def _paired_accuracy_interval(reference, candidate, targets, *, seed=SEED, repetitions=5000):
    ref_ok = reference.argmax(axis=1) == targets
    cand_ok = candidate.argmax(axis=1) == targets
    delta = cand_ok.astype(np.float64) - ref_ok.astype(np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, delta.size, size=(repetitions, delta.size))
    values = delta[samples].mean(axis=1)
    return {
        "delta_accuracy": float(delta.mean()),
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": seed,
        "ci_95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "resampling_unit": "video group; one row per video",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    manifest_path = args.manifest.resolve(strict=True)
    video_path = args.video_features.resolve(strict=True)
    audio_path = args.audio_features.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite readout output: {output}")
    manifest, dataset = load_dataset(manifest_path, video_path, audio_path)
    train_rows, valid_rows = dataset["train"], dataset["validation"]
    train_video_raw = np.stack([row["video"] for row in train_rows])
    train_audio_raw = np.stack([row["audio"] for row in train_rows])
    video_mean = train_video_raw.reshape(-1, train_video_raw.shape[-1]).mean(axis=0)
    video_std = train_video_raw.reshape(-1, train_video_raw.shape[-1]).std(axis=0).clip(1e-5)
    audio_mean = train_audio_raw.reshape(-1, train_audio_raw.shape[-1]).mean(axis=0)
    audio_std = train_audio_raw.reshape(-1, train_audio_raw.shape[-1]).std(axis=0).clip(1e-5)
    train = _arrays(train_rows, video_mean, video_std, audio_mean, audio_std)
    validation = _arrays(valid_rows, video_mean, video_std, audio_mean, audio_std)
    output.mkdir(parents=True)
    np.savez_compressed(
        output / "train-normalization.npz",
        video_mean=video_mean,
        video_std=video_std,
        audio_mean=audio_mean,
        audio_std=audio_std,
    )
    config = {
        "seed": SEED,
        "total_updates_per_arm": TOTAL_UPDATES,
        "evaluation_interval": EVALUATION_INTERVAL,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "hidden_dimension": HIDDEN_DIMENSION,
        "optimizer": "AdamW",
        "checkpoint_selector": "minimum validation categorical NLL; ties keep earlier step",
        "normalization": "train-only per-feature mean/std; std floor=1e-5",
        "video_task": manifest["task_id"],
        "source_split": "official train only; internal video-group holdout",
        "validation_is_blind": False,
    }
    (output / "effective_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    history_path = output / "training_history.jsonl"
    started = time.perf_counter()
    arms = {}
    factories = (
        ("audio_only", lambda: AudioCandidateScorer()),
        ("pooled_av", lambda: TemporalCandidateScorer(pool_video=True)),
        ("sequence_av", lambda: TemporalCandidateScorer(pool_video=False)),
    )
    for name, factory in factories:
        torch.manual_seed(SEED)
        model = factory()
        arms[name] = _fit_arm(
            name,
            model,
            (None, train[1], train[2]) if name == "audio_only" else train,
            (None, validation[1], validation[2]) if name == "audio_only" else validation,
            train_rows,
            valid_rows,
            output,
            history_path,
        )

    valid_targets = validation[2].numpy()
    sequence_model = arms["sequence_av"]["model"]
    valid_video, valid_audio = validation[0], validation[1]
    base_sequence_scores = _predict(sequence_model, valid_video, valid_audio)
    controls = {}
    for name, video_values, audio_values in (
        ("video_missing", torch.zeros_like(valid_video), valid_audio),
        ("audio_missing", valid_video, torch.zeros_like(valid_audio)),
        ("time_reversed", valid_video.flip(dims=(1,)), valid_audio),
    ):
        scores = _predict(sequence_model, video_values, audio_values)
        controls[name] = {
            "metrics": {
                k: v
                for k, v in categorical_metrics(scores, valid_targets).items()
                if k != "probabilities"
            },
            "paired_accuracy_delta_vs_sequence": _paired_accuracy_interval(
                base_sequence_scores, scores, valid_targets
            ),
        }
    video_only_probabilities = np.full((len(valid_rows), 2), 0.5, dtype=np.float64)
    summary = {
        "status": "PASS_FRESH_TEMPORAL_READOUT_DEVELOPMENT_FIT",
        "manifest_sha256": sha256_file(manifest_path),
        "video_feature_result_sha256": sha256_file(video_path),
        "audio_feature_result_sha256": sha256_file(audio_path),
        "task_id": manifest["task_id"],
        "train_video_groups": len(train_rows),
        "validation_video_groups": len(valid_rows),
        "train_validation_overlap": 0,
        "arms": {
            name: {k: v for k, v in result.items() if k not in {"model", "validation_scores"}}
            for name, result in arms.items()
        },
        "video_only_uniform_tie_baseline": {
            "metrics": {
                k: v
                for k, v in categorical_metrics(
                    np.zeros_like(video_only_probabilities), valid_targets
                ).items()
                if k != "probabilities"
            },
            "probabilities": video_only_probabilities.tolist(),
            "interpretation": (
                "The same video sequence is presented for both audio options; "
                "video-only cannot distinguish the option identity."
            ),
        },
        "sequence_input_controls": controls,
        "paired_accuracy_deltas_vs_audio_only": _paired_accuracy_interval(
            arms["audio_only"]["validation_scores"],
            arms["sequence_av"]["validation_scores"],
            valid_targets,
        ),
        "paired_accuracy_deltas_sequence_vs_pooled": _paired_accuracy_interval(
            arms["pooled_av"]["validation_scores"],
            arms["sequence_av"]["validation_scores"],
            valid_targets,
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "validation_scope": (
            "fresh development holdout from official train; checkpoint selection "
            "used these rows; not blind or independent test"
        ),
    }
    predictions = []
    for index, row in enumerate(valid_rows):
        predictions.append(
            {
                "sample_id": row["sample_id"],
                "video_id": row["video_id"],
                "target_index": row["target"],
                "arms": {
                    name: {
                        "scores": result["validation_scores"][index].tolist(),
                        "probabilities": categorical_metrics(
                            result["validation_scores"][index : index + 1],
                            np.asarray([row["target"]]),
                        )["probabilities"][0].tolist(),
                    }
                    for name, result in arms.items()
                },
            }
        )
    (output / "validation_predictions.json").write_text(
        json.dumps(predictions, indent=2) + "\n", encoding="utf-8"
    )
    (output / "result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-features", type=Path, required=True)
    parser.add_argument("--audio-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
