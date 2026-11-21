"""Nested recording-stratified CV using only SoundActions train recordings."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    from train_soundactions_temporal_readout import CLASS_TO_ID, _load_features
except ModuleNotFoundError:
    from scripts.train_soundactions_temporal_readout import CLASS_TO_ID, _load_features
from tiny_omni_decision.soundactions_metrics import (
    multiclass_metrics,
    paired_bootstrap_difference,
)

SEED = 17
CONDITIONS = ("audio", "video", "gated_av")
MAX_EPOCHS = 60
PATIENCE = 12
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_SIZE = 32


def stratified_folds(
    targets: np.ndarray,
    *,
    group_ids: np.ndarray | None = None,
    fold_count: int = 5,
    seed: int = SEED,
) -> np.ndarray:
    """Assign every row once, balancing labels without consulting holdout data."""
    labels = np.asarray(targets, dtype=np.int64)
    if labels.ndim != 1 or fold_count < 2 or len(labels) < fold_count:
        raise ValueError("targets must be a vector large enough for at least two folds")
    groups = np.arange(len(labels)) if group_ids is None else np.asarray(group_ids)
    if groups.ndim != 1 or len(groups) != len(labels):
        raise ValueError("group IDs must align one-to-one with targets")
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    group_labels = np.full(len(unique_groups), -1, dtype=np.int64)
    for group_index in range(len(unique_groups)):
        observed_labels = np.unique(labels[inverse == group_index])
        if len(observed_labels) != 1:
            raise ValueError("all rows from one group must have the same target")
        group_labels[group_index] = observed_labels[0]
    group_assignment = np.full(len(unique_groups), -1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    fold_offsets = np.zeros(fold_count, dtype=np.int64)
    for label in np.unique(group_labels):
        indices = np.flatnonzero(group_labels == label)
        rng.shuffle(indices)
        start = int(np.argmin(fold_offsets))
        ordered_folds = (np.arange(len(indices)) + start) % fold_count
        group_assignment[indices] = ordered_folds
        counts = np.bincount(ordered_folds, minlength=fold_count)
        fold_offsets += counts
    assignment = group_assignment[inverse]
    if np.any(assignment < 0) or set(np.unique(assignment)) != set(range(fold_count)):
        raise RuntimeError("fold assignment did not cover every row and fold")
    for fold in range(fold_count):
        if len(np.unique(labels[assignment == fold])) != len(np.unique(labels)):
            raise ValueError("each fold must contain every target class")
    return assignment


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stratified_inner_split(
    indices: np.ndarray, targets: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    fit_parts, stop_parts = [], []
    for label in np.unique(targets[indices]):
        selected = indices[targets[indices] == label].copy()
        rng.shuffle(selected)
        stop_count = max(1, int(round(len(selected) * 0.2)))
        stop_parts.append(selected[:stop_count])
        fit_parts.append(selected[stop_count:])
    return np.sort(np.concatenate(fit_parts)), np.sort(np.concatenate(stop_parts))


def _fit_outer_fold(
    condition: str,
    data: dict,
    fit_indices: np.ndarray,
    stop_indices: np.ndarray,
    test_indices: np.ndarray,
    torch,
    *,
    seed: int,
) -> dict[str, object]:
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence
    from torch.utils.data import DataLoader, Dataset

    class RecordDataset(Dataset):
        def __init__(self, indices):
            self.indices = np.asarray(indices, dtype=np.int64)

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, item):
            index = int(self.indices[item])
            first, last = data["audio_offsets"][index : index + 2]
            return (
                index,
                data["audio"][first:last],
                data["video"][index],
                int(data["targets"][index]),
            )

    def collate(rows):
        audio = [torch.from_numpy(row[1]) for row in rows]
        lengths = torch.tensor([len(item) for item in audio], dtype=torch.int64)
        return (
            np.asarray([row[0] for row in rows], dtype=np.int64),
            nn.utils.rnn.pad_sequence(audio, batch_first=True),
            lengths,
            torch.from_numpy(np.stack([row[2] for row in rows])),
            torch.tensor([row[3] for row in rows], dtype=torch.long),
        )

    class TemporalEncoder(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.recurrent = nn.LSTM(input_size, HIDDEN_SIZE, batch_first=True, bidirectional=True)
            self.norm = nn.LayerNorm(HIDDEN_SIZE * 2)

        def forward(self, sequence, lengths):
            packed = pack_padded_sequence(
                sequence, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (hidden, _) = self.recurrent(packed)
            return self.norm(torch.cat((hidden[-2], hidden[-1]), dim=1))

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            if condition in {"audio", "gated_av"}:
                self.audio_encoder = TemporalEncoder(384)
            if condition in {"video", "gated_av"}:
                self.video_encoder = TemporalEncoder(768)
            if condition == "gated_av":
                self.gate = nn.Sequential(
                    nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 64), nn.Sigmoid()
                )
            self.dropout = nn.Dropout(0.25)
            self.classifier = nn.Linear(64, len(CLASS_TO_ID))

        def forward(self, audio, lengths, video):
            audio_repr = (
                self.audio_encoder(audio, lengths) if condition in {"audio", "gated_av"} else None
            )
            video_repr = None
            if condition in {"video", "gated_av"}:
                video_lengths = torch.full((len(video),), video.shape[1], dtype=torch.int64)
                video_repr = self.video_encoder(video, video_lengths)
            if condition == "audio":
                fused = audio_repr
            elif condition == "video":
                fused = video_repr
            else:
                gate = self.gate(torch.cat((audio_repr, video_repr), dim=1))
                fused = gate * audio_repr + (1.0 - gate) * video_repr
            return self.classifier(self.dropout(fused))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator(device="cpu").manual_seed(seed)
    fit_loader = DataLoader(
        RecordDataset(fit_indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        collate_fn=collate,
    )
    stop_loader = DataLoader(
        RecordDataset(stop_indices), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate
    )
    test_loader = DataLoader(
        RecordDataset(test_indices), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate
    )
    model = Classifier().cpu()
    params = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best_nll, best_epoch, stale, best_state = float("inf"), 0, 0, None
    history = []
    started = time.perf_counter()

    def predict(loader, *, train=False):
        model.train(train)
        all_indices, all_logits, all_targets = [], [], []
        with torch.set_grad_enabled(train):
            for indices, audio, lengths, video, targets in loader:
                logits = model(audio, lengths, video)
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    nn.functional.cross_entropy(logits, targets).backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                all_indices.extend(indices.tolist())
                all_logits.append(logits.detach().cpu())
                all_targets.append(targets)
        logits = torch.cat(all_logits)
        targets = torch.cat(all_targets).numpy()
        order = np.argsort(np.asarray(all_indices))
        return (
            np.asarray(all_indices)[order],
            torch.softmax(logits, dim=1).numpy()[order],
            targets[order],
        )

    for epoch in range(1, MAX_EPOCHS + 1):
        predict(fit_loader, train=True)
        _, stop_prob, stop_targets = predict(stop_loader)
        stop_nll = multiclass_metrics(stop_prob, stop_targets)["nll"]
        history.append({"epoch": epoch, "inner_stop_nll": stop_nll})
        if stop_nll < best_nll:
            best_nll, best_epoch, stale = stop_nll, epoch, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("inner selection did not save a checkpoint")
    model.load_state_dict(best_state)
    test_order, probabilities, targets = predict(test_loader)
    elapsed = time.perf_counter() - started
    return {
        "condition": condition,
        "seed": seed,
        "trainable_parameters": params,
        "fit_count": len(fit_indices),
        "inner_stop_count": len(stop_indices),
        "outer_test_count": len(test_indices),
        "selected_epoch": best_epoch,
        "stopped_epoch": len(history),
        "selected_inner_stop_nll": best_nll,
        "outer_test_indices": test_order.tolist(),
        "outer_test_targets": targets.tolist(),
        "outer_test_probabilities": probabilities.tolist(),
        "outer_test_metrics": multiclass_metrics(probabilities, targets),
        "history": history,
        "wall_seconds": elapsed,
    }


def run_cv(
    audio_dir: Path, video_dir: Path, output: Path, *, fold_count: int = 5
) -> dict[str, object]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    data = _load_features(audio_dir, video_dir)
    indices = np.flatnonzero(data["splits"] == "train")
    targets = data["targets"]
    folds = stratified_folds(
        targets[indices], group_ids=data["ids"][indices], fold_count=fold_count, seed=SEED
    )
    fold_for_global = np.full(len(targets), -1, dtype=np.int64)
    fold_for_global[indices] = folds
    run_rows = []
    all_predictions = {
        condition: np.full((len(indices), len(CLASS_TO_ID)), np.nan) for condition in CONDITIONS
    }
    index_to_train_pos = {int(index): pos for pos, index in enumerate(indices)}
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("train-only CV must run in a fresh process without CUDA initialized")
    for fold in range(fold_count):
        test_indices = indices[folds == fold]
        outer_train = indices[folds != fold]
        fit_indices, stop_indices = _stratified_inner_split(outer_train, targets, SEED + 100 + fold)
        if set(fit_indices) & set(stop_indices) or set(outer_train) & set(test_indices):
            raise RuntimeError("inner or outer recording split overlap")
        fold_seed = SEED + 1000 + fold
        for condition in CONDITIONS:
            result = _fit_outer_fold(
                condition, data, fit_indices, stop_indices, test_indices, torch, seed=fold_seed
            )
            predicted_indices = result["outer_test_indices"]
            expected = test_indices.tolist()
            if set(predicted_indices) != set(expected) or len(predicted_indices) != len(
                set(predicted_indices)
            ):
                raise RuntimeError("outer predictions do not cover their fold exactly once")
            for index, probability in zip(
                predicted_indices, result["outer_test_probabilities"], strict=True
            ):
                all_predictions[condition][index_to_train_pos[index]] = probability
            run_rows.append({"fold": fold, **result})
            (output / f"fold-{fold}-{condition}.json").write_text(
                json.dumps({"fold": fold, **result}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    if any(not np.isfinite(probabilities).all() for probabilities in all_predictions.values()):
        raise RuntimeError("out-of-fold predictions are incomplete")
    train_targets = targets[indices]
    aggregate = {
        condition: multiclass_metrics(probabilities, train_targets)
        for condition, probabilities in all_predictions.items()
    }
    comparisons = {
        "gated_av_vs_audio_accuracy": paired_bootstrap_difference(
            all_predictions["audio"],
            all_predictions["gated_av"],
            train_targets,
            metric="accuracy",
            seed=SEED,
            iterations=10_000,
        ),
        "gated_av_vs_audio_nll": paired_bootstrap_difference(
            all_predictions["audio"],
            all_predictions["gated_av"],
            train_targets,
            metric="nll",
            seed=SEED,
            iterations=10_000,
        ),
        "gated_av_vs_video_accuracy": paired_bootstrap_difference(
            all_predictions["video"],
            all_predictions["gated_av"],
            train_targets,
            metric="accuracy",
            seed=SEED,
            iterations=10_000,
        ),
        "gated_av_vs_video_nll": paired_bootstrap_difference(
            all_predictions["video"],
            all_predictions["gated_av"],
            train_targets,
            metric="nll",
            seed=SEED,
            iterations=10_000,
        ),
    }
    summary = {
        "status": "COMPLETE_TRAIN_ONLY_NESTED_CV",
        "protocol": {
            "seed": SEED,
            "outer_folds": fold_count,
            "split_unit": "recording ID",
            "inner_stop_fraction_per_class": 0.2,
            "inner_selection": "lowest inner-stop NLL; exact tie keeps earlier epoch",
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "hidden_size_per_direction": HIDDEN_SIZE,
            "evaluation_scope": "train recordings only; validation/audit excluded",
            "bootstrap": "10k paired class-stratified; approximate with shared fold training",
        },
        "data": {
            "train_recording_ids": data["ids"][indices].astype(int).tolist(),
            "train_id_sha256": hashlib.sha256(
                np.asarray(data["ids"][indices], dtype="<i8").tobytes()
            ).hexdigest(),
            "labels_by_class": {
                label: int(np.sum(targets[indices] == class_id))
                for label, class_id in CLASS_TO_ID.items()
            },
            "audio_feature_cache_sha256": data["audio_cache_sha256"],
            "video_feature_cache_sha256": data["video_cache_sha256"],
            "fold_assignment": folds.tolist(),
            "fold_assignment_sha256": hashlib.sha256(folds.astype("<i8").tobytes()).hexdigest(),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": "cpu",
            "cuda_initialized": torch.cuda.is_initialized(),
            "source_tree_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
        "conditions": aggregate,
        "comparisons": comparisons,
        "per_fold": [
            {
                "fold": row["fold"],
                "condition": row["condition"],
                "selected_epoch": row["selected_epoch"],
                "stopped_epoch": row["stopped_epoch"],
                "outer_test_metrics": row["outer_test_metrics"],
                "wall_seconds": row["wall_seconds"],
            }
            for row in run_rows
        ],
        "files": {},
    }
    prediction_payload = {
        "train_ids": data["ids"][indices].astype(int).tolist(),
        "targets": train_targets.astype(int).tolist(),
        "probabilities": {key: value.tolist() for key, value in all_predictions.items()},
    }
    prediction_path = output / "out-of-fold-predictions.json"
    prediction_path.write_text(
        json.dumps(prediction_payload, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary["files"][prediction_path.name] = _sha256(prediction_path)
    summary["files"].update(
        {path.name: _sha256(path) for path in sorted(output.glob("fold-*.json"))}
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-features", type=Path, required=True)
    parser.add_argument("--video-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    if args.folds != 5:
        raise ValueError("the preregistered analysis uses exactly five outer folds")
    result = run_cv(args.audio_features, args.video_features, args.output, fold_count=args.folds)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output),
                "conditions": result["conditions"],
                "comparisons": result["comparisons"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
