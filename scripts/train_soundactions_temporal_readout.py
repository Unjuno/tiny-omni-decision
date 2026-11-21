"""Fit audio-only, video-only, and gated temporal Decision readouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from tiny_omni_decision.soundactions_metrics import (
    multiclass_metrics,
    paired_bootstrap_difference,
)

SEED = 17
MAX_EPOCHS = 500
PATIENCE = 100
MIN_DELTA = 0.0
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
HIDDEN_SIZE = 32
CLASS_TO_ID = {"impulsive": 0, "sustained": 1, "iterative": 2}
CONDITIONS = ("audio", "video", "gated_av")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validation_improved(score: float, best_score: float, min_delta: float = MIN_DELTA) -> bool:
    return score < best_score - min_delta


def should_stop(epoch: int, best_epoch: int, patience: int = PATIENCE) -> bool:
    if epoch < 1 or best_epoch < 1 or patience < 1:
        raise ValueError("epoch, best epoch, and patience must be positive")
    return epoch - best_epoch >= patience


def _load_features(audio_dir: Path, video_dir: Path):
    audio_report_path = audio_dir / "feature-report.json"
    video_report_path = video_dir / "feature-report.json"
    audio_report = json.loads(audio_report_path.read_text(encoding="utf-8"))
    video_report = json.loads(video_report_path.read_text(encoding="utf-8"))
    if audio_report.get("status") != "PASS_FROZEN_AUDIO_FEATURES":
        raise ValueError("audio feature report did not pass")
    if video_report.get("status") != "PASS_FROZEN_VIDEO_FEATURES":
        raise ValueError("video feature report did not pass")
    audio_path = audio_dir / str(audio_report["feature_cache"])
    video_path = video_dir / str(video_report["feature_cache"])
    if _sha256_file(audio_path) != audio_report["feature_cache_sha256"]:
        raise ValueError("audio feature cache hash mismatch")
    if _sha256_file(video_path) != video_report["feature_cache_sha256"]:
        raise ValueError("video feature cache hash mismatch")
    with np.load(audio_path, allow_pickle=False) as archive:
        audio = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(video_path, allow_pickle=False) as archive:
        video = {key: np.asarray(archive[key]) for key in archive.files}
    for key in ("recording_ids", "splits", "labels"):
        if not np.array_equal(audio[key], video[key]):
            raise ValueError(f"audio/video split or label metadata differs: {key}")
    ids = audio["recording_ids"].astype(np.int64)
    splits = audio["splits"].astype(str)
    labels = audio["labels"].astype(str)
    if len(ids) != len(set(ids.tolist())) or any(
        split not in {"train", "validation"} for split in splits
    ):
        raise ValueError("feature cache has duplicate IDs or a forbidden split")
    if any(label not in CLASS_TO_ID for label in labels):
        raise ValueError("feature cache contains an unsupported task label")
    audio_features = np.asarray(audio["features"], dtype=np.float32)
    offsets = np.asarray(audio["offsets"], dtype=np.int64)
    video_features = np.asarray(video["features"], dtype=np.float32)
    if (
        audio_features.ndim != 2
        or audio_features.shape[1] != 384
        or offsets.shape != (len(ids) + 1,)
        or offsets[0] != 0
        or offsets[-1] != len(audio_features)
        or np.any(np.diff(offsets) < 1)
        or video_features.shape != (len(ids), 8, 768)
        or not np.isfinite(audio_features).all()
        or not np.isfinite(video_features).all()
    ):
        raise ValueError("feature sequences have an invalid shape, length, or value")
    return {
        "ids": ids,
        "splits": splits,
        "labels": labels,
        "targets": np.asarray([CLASS_TO_ID[label] for label in labels], dtype=np.int64),
        "audio": audio_features,
        "audio_offsets": offsets,
        "video": video_features,
        "audio_report": audio_report,
        "video_report": video_report,
        "audio_cache_sha256": audio_report["feature_cache_sha256"],
        "video_cache_sha256": video_report["feature_cache_sha256"],
    }


def _seed_everything(torch, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _fit_condition(condition: str, data, output: Path, torch, device) -> dict[str, object]:
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence
    from torch.utils.data import DataLoader, Dataset

    class Records(Dataset):
        def __init__(self, indices):
            self.indices = np.asarray(indices, dtype=np.int64)

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, item):
            index = int(self.indices[item])
            first, last = data["audio_offsets"][index : index + 2]
            return {
                "index": index,
                "audio": data["audio"][first:last],
                "video": data["video"][index],
                "target": int(data["targets"][index]),
            }

    def collate(rows):
        audio_sequences = [torch.from_numpy(row["audio"]) for row in rows]
        audio_lengths = torch.tensor(
            [len(sequence) for sequence in audio_sequences], dtype=torch.int64
        )
        audio_padded = nn.utils.rnn.pad_sequence(audio_sequences, batch_first=True)
        video = torch.from_numpy(np.stack([row["video"] for row in rows]))
        targets = torch.tensor([row["target"] for row in rows], dtype=torch.long)
        indices = np.asarray([row["index"] for row in rows], dtype=np.int64)
        return audio_padded, audio_lengths, video, targets, indices

    class TemporalEncoder(nn.Module):
        def __init__(self, input_size: int):
            super().__init__()
            self.recurrent = nn.LSTM(
                input_size=input_size,
                hidden_size=HIDDEN_SIZE,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.norm = nn.LayerNorm(HIDDEN_SIZE * 2)

        def forward(self, sequence, lengths):
            packed = pack_padded_sequence(
                sequence, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (hidden, _) = self.recurrent(packed)
            return self.norm(torch.cat((hidden[-2], hidden[-1]), dim=1))

    class TemporalClassifier(nn.Module):
        def __init__(self, mode: str):
            super().__init__()
            self.mode = mode
            if mode in {"audio", "gated_av"}:
                self.audio_encoder = TemporalEncoder(384)
            if mode in {"video", "gated_av"}:
                self.video_encoder = TemporalEncoder(768)
            self.dropout = nn.Dropout(0.25)
            if mode == "gated_av":
                self.gate = nn.Sequential(
                    nn.Linear(HIDDEN_SIZE * 4, HIDDEN_SIZE * 2),
                    nn.Tanh(),
                    nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE * 2),
                    nn.Sigmoid(),
                )
            self.classifier = nn.Linear(HIDDEN_SIZE * 2, len(CLASS_TO_ID))

        def forward(self, audio_seq, audio_len, video_seq, *, zero_audio=False, zero_video=False):
            audio_repr = None
            video_repr = None
            if self.mode in {"audio", "gated_av"}:
                audio_input = torch.zeros_like(audio_seq) if zero_audio else audio_seq
                audio_repr = self.audio_encoder(audio_input, audio_len)
            if self.mode in {"video", "gated_av"}:
                video_input = torch.zeros_like(video_seq) if zero_video else video_seq
                lengths = torch.full((len(video_input),), video_input.shape[1], dtype=torch.int64)
                video_repr = self.video_encoder(video_input, lengths)
            gate = None
            if self.mode == "audio":
                fused = audio_repr
            elif self.mode == "video":
                fused = video_repr
            else:
                gate = self.gate(torch.cat((audio_repr, video_repr), dim=1))
                fused = gate * audio_repr + (1.0 - gate) * video_repr
            return self.classifier(self.dropout(fused)), gate

    train_indices = np.flatnonzero(data["splits"] == "train")
    valid_indices = np.flatnonzero(data["splits"] == "validation")
    if len(train_indices) < 30 or len(valid_indices) < 10:
        raise ValueError("too few labeled recordings for the frozen train/validation experiment")
    condition_dir = output / condition
    condition_dir.mkdir(parents=True)
    train_loader = DataLoader(
        Records(train_indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        generator=torch.Generator().manual_seed(SEED),
    )
    train_eval_loader = DataLoader(
        Records(train_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )
    valid_loader = DataLoader(
        Records(valid_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    _seed_everything(torch, SEED)
    model = TemporalClassifier(condition).to(device)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    best_score = float("inf")
    best_epoch = 0
    best_metrics = None
    best_state = None
    history_path = condition_dir / "training-history.jsonl"
    total_updates = 0
    train_seconds = 0.0
    evaluation_seconds = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    run_started = time.perf_counter()

    def evaluate(
        loader,
        *,
        capture=False,
        zero_audio=False,
        zero_video=False,
        video_override=None,
    ):
        model.eval()
        truth_rows, prob_rows, ids_rows = [], [], []
        gate_rows = []
        with torch.inference_mode():
            for audio_seq, audio_len, video_seq, targets, indices in loader:
                if video_override is not None:
                    video_seq = torch.from_numpy(video_override[indices])
                logits, gate = model(
                    audio_seq.to(device),
                    audio_len,
                    video_seq.to(device),
                    zero_audio=zero_audio,
                    zero_video=zero_video,
                )
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()
                truth_rows.extend(targets.numpy().tolist())
                prob_rows.extend(probabilities.tolist())
                ids_rows.extend(indices.tolist())
                if gate is not None:
                    gate_rows.extend(gate.cpu().numpy().tolist())
        probabilities = np.asarray(prob_rows, dtype=np.float64)
        targets = np.asarray(truth_rows, dtype=np.int64)
        metrics = multiclass_metrics(probabilities, targets)
        if capture:
            return metrics, probabilities, targets, np.asarray(ids_rows), np.asarray(gate_rows)
        return metrics

    for epoch in range(1, MAX_EPOCHS + 1):
        train_started = time.perf_counter()
        model.train()
        epoch_loss = 0.0
        epoch_examples = 0
        for audio_seq, audio_len, video_seq, targets, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(audio_seq.to(device), audio_len, video_seq.to(device))
            loss = criterion(logits, targets.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite training loss at {condition} epoch {epoch}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"nonfinite gradient norm at {condition} epoch {epoch}")
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(targets)
            epoch_examples += len(targets)
            total_updates += 1
        train_seconds += time.perf_counter() - train_started

        eval_started = time.perf_counter()
        fixed_train_metrics = evaluate(train_eval_loader)
        (
            validation_metrics,
            validation_probabilities,
            validation_targets,
            validation_order,
            gate_rows,
        ) = evaluate(valid_loader, capture=True)
        evaluation_seconds += time.perf_counter() - eval_started
        selected_score = float(validation_metrics["nll"])
        improved = validation_improved(selected_score, best_score)
        if improved:
            best_score = selected_score
            best_epoch = epoch
            best_metrics = validation_metrics
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            torch.save(
                {
                    "condition": condition,
                    "seed": SEED,
                    "epoch": epoch,
                    "model_state_dict": best_state,
                    "trainable_parameters": trainable_parameters,
                    "validation_metrics": validation_metrics,
                },
                condition_dir / "best.pt",
            )
            prediction_path = condition_dir / "best-validation-predictions.jsonl"
            with prediction_path.open("w", encoding="utf-8", newline="\n") as stream:
                for index, truth, probs in zip(
                    validation_order, validation_targets, validation_probabilities, strict=True
                ):
                    stream.write(
                        json.dumps(
                            {
                                "recording_id": int(data["ids"][index]),
                                "target": list(CLASS_TO_ID)[int(truth)],
                                "prediction": list(CLASS_TO_ID)[int(np.argmax(probs))],
                                "probabilities": probs.tolist(),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
        history_row = {
            "epoch": epoch,
            "optimizer_updates": total_updates,
            "examples_seen_this_epoch": epoch_examples,
            "examples_seen_cumulative": epoch * len(train_indices),
            "train_ce_rolling": epoch_loss / max(epoch_examples, 1),
            "fixed_train_metrics": fixed_train_metrics,
            "validation_metrics": validation_metrics,
            "selected_checkpoint_nll": best_score,
            "best_epoch": best_epoch,
            "improved": improved,
            "gate_mean": float(gate_rows.mean()) if len(gate_rows) else None,
            "gate_std": float(gate_rows.std()) if len(gate_rows) else None,
            "epoch_wall_seconds": time.perf_counter() - train_started,
        }
        with history_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(history_row, sort_keys=True) + "\n")
        if epoch == 1 or improved or epoch % 10 == 0:
            print(
                json.dumps(
                    {
                        "condition": condition,
                        "epoch": epoch,
                        "train_nll": fixed_train_metrics["nll"],
                        "validation_nll": validation_metrics["nll"],
                        "validation_accuracy": validation_metrics["accuracy"],
                        "best_epoch": best_epoch,
                    }
                ),
                flush=True,
            )
        if should_stop(epoch, best_epoch):
            break

    final_epoch = epoch
    torch.save(
        {
            "condition": condition,
            "seed": SEED,
            "epoch": final_epoch,
            "model_state_dict": model.state_dict(),
            "trainable_parameters": trainable_parameters,
        },
        condition_dir / "final.pt",
    )
    final_train_metrics = evaluate(train_eval_loader)
    final_validation_metrics = evaluate(valid_loader)
    if best_state is None:
        raise RuntimeError(f"{condition} never produced a validation-selected checkpoint")
    best_checkpoint = torch.load(condition_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    selected_train_metrics = evaluate(train_eval_loader)
    (
        selected_valid_metrics,
        selected_probabilities,
        selected_targets,
        selected_order,
        selected_gates,
    ) = evaluate(valid_loader, capture=True)
    if selected_valid_metrics != best_metrics:
        for key in ("accuracy", "nll", "brier", "ece"):
            if abs(float(selected_valid_metrics[key]) - float(best_metrics[key])) > 1e-8:
                raise ValueError(f"best checkpoint reload changed validation {key}")
    with (condition_dir / "selected-validation-predictions.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        for index, truth, probs in zip(
            selected_order, selected_targets, selected_probabilities, strict=True
        ):
            stream.write(
                json.dumps(
                    {
                        "recording_id": int(data["ids"][index]),
                        "target": list(CLASS_TO_ID)[int(truth)],
                        "prediction": list(CLASS_TO_ID)[int(np.argmax(probs))],
                        "probabilities": probs.tolist(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    controls = None
    if condition == "gated_av":
        rng = np.random.default_rng(SEED)
        permutation = np.arange(len(valid_indices))
        for index in range(len(permutation) - 1, 0, -1):
            donor = int(rng.integers(0, index))
            permutation[index], permutation[donor] = permutation[donor], permutation[index]
        mismatched_video = data["video"].copy()
        mismatched_video[valid_indices] = data["video"][valid_indices[permutation]]
        controls = {
            "validation_video_identity_permuted": evaluate(
                valid_loader, video_override=mismatched_video
            ),
            "validation_audio_zero_substitution": evaluate(valid_loader, zero_audio=True),
            "validation_video_zero_substitution": evaluate(valid_loader, zero_video=True),
            "video_permutation_seed": SEED,
            "video_permutation_derangement": bool(
                np.all(permutation != np.arange(len(valid_indices)))
            ),
        }
    report = {
        "status": "PASS",
        "condition": condition,
        "configuration": {
            "seed": SEED,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "min_delta": MIN_DELTA,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRAD_CLIP_NORM,
            "bilstm_hidden_size_per_direction": HIDDEN_SIZE,
            "dropout": 0.25,
            "class_order": list(CLASS_TO_ID),
        },
        "seed": SEED,
        "selected_epoch": best_epoch,
        "final_epoch": final_epoch,
        "optimizer_updates": total_updates,
        "trainable_parameters": trainable_parameters,
        "train_count": len(train_indices),
        "validation_count": len(valid_indices),
        "selected_train_metrics": selected_train_metrics,
        "selected_validation_metrics": selected_valid_metrics,
        "final_train_metrics": final_train_metrics,
        "final_validation_metrics": final_validation_metrics,
        "train_validation_nll_gap": float(selected_valid_metrics["nll"])
        - float(selected_train_metrics["nll"]),
        "early_stopping_patience": PATIENCE,
        "best_checkpoint_rule": "lowest validation NLL, exact tie keeps earlier epoch",
        "best_checkpoint_sha256": _sha256_file(condition_dir / "best.pt"),
        "best_checkpoint_bytes": (condition_dir / "best.pt").stat().st_size,
        "final_checkpoint_sha256": _sha256_file(condition_dir / "final.pt"),
        "final_checkpoint_bytes": (condition_dir / "final.pt").stat().st_size,
        "train_wall_seconds": train_seconds,
        "evaluation_wall_seconds": evaluation_seconds,
        "wall_seconds": time.perf_counter() - run_started,
        "peak_allocated_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "device": str(device),
        "overfitting_signal": (
            float(final_train_metrics["nll"]) < float(selected_train_metrics["nll"])
            and float(final_validation_metrics["nll"]) > float(selected_valid_metrics["nll"])
        ),
        "gate_mean": float(selected_gates.mean()) if len(selected_gates) else None,
        "gate_std": float(selected_gates.std()) if len(selected_gates) else None,
        "validation_controls": controls,
    }
    (condition_dir / "result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        **report,
        "validation_probabilities": selected_probabilities,
        "validation_targets": selected_targets,
        "validation_order": selected_order,
    }


def run(audio_dir: Path, video_dir: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment output: {output}")
    data = _load_features(audio_dir.resolve(strict=True), video_dir.resolve(strict=True))
    output.mkdir(parents=True)
    import torch

    _seed_everything(torch, SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for condition in CONDITIONS:
        results[condition] = _fit_condition(condition, data, output, torch, device)
    validation_mask = data["splits"][results["gated_av"]["validation_order"]] == "validation"
    if not validation_mask.all():
        raise ValueError("selected validation order unexpectedly includes a train recording")
    probabilities = {key: results[key]["validation_probabilities"] for key in CONDITIONS}
    targets = results["gated_av"]["validation_targets"]
    ids = data["ids"][results["gated_av"]["validation_order"]]
    for key in CONDITIONS:
        if not np.array_equal(
            results[key]["validation_order"], results["gated_av"]["validation_order"]
        ):
            raise ValueError("condition validation prediction order differs")

    comparisons = {
        "validation_selection_affected": True,
        "bootstrap": {
            f"gated_av_vs_{other}_{metric}": paired_bootstrap_difference(
                probabilities[other],
                probabilities["gated_av"],
                targets,
                metric=metric,
                seed=SEED,
                iterations=10_000,
            )
            for other in ("audio", "video")
            for metric in ("accuracy", "macro_accuracy", "nll", "brier")
        },
        "condition_metrics": {
            key: results[key]["selected_validation_metrics"] for key in CONDITIONS
        },
        "gated_av_controls": results["gated_av"]["validation_controls"],
        "validation_recording_ids": ids.astype(int).tolist(),
        "class_order": list(CLASS_TO_ID),
        "seed": SEED,
        "bootstrap_resamples": 10_000,
    }
    (output / "comparisons.json").write_text(
        json.dumps(comparisons, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "PASS",
        "seed": SEED,
        "command_argv": [
            "python",
            "-m",
            "scripts.train_soundactions_temporal_readout",
            *sys.argv[1:],
        ],
        "source_commit": subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "training_configuration": {
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "min_delta": MIN_DELTA,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRAD_CLIP_NORM,
            "bilstm_hidden_size_per_direction": HIDDEN_SIZE,
            "dropout": 0.25,
            "best_checkpoint_rule": "lowest validation NLL, exact tie keeps earlier epoch",
        },
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "conditions": {
            key: {
                "selected_epoch": results[key]["selected_epoch"],
                "final_epoch": results[key]["final_epoch"],
                "selected_validation_metrics": results[key]["selected_validation_metrics"],
                "train_validation_nll_gap": results[key]["train_validation_nll_gap"],
                "trainable_parameters": results[key]["trainable_parameters"],
                "peak_allocated_vram_bytes": results[key]["peak_allocated_vram_bytes"],
            }
            for key in CONDITIONS
        },
        "feature_cache_sha256": {
            "audio": data["audio_cache_sha256"],
            "video": data["video_cache_sha256"],
        },
        "split_ids_sha256": {
            split: hashlib.sha256(
                json.dumps(
                    data["ids"][data["splits"] == split].astype(int).tolist(),
                    separators=(",", ":"),
                ).encode("ascii")
            ).hexdigest()
            for split in ("train", "validation")
        },
        "frozen_encoder_parameters": {
            "audio": 1_961_239,
            "video": 114_365_184,
            "paired_union": 116_326_423,
        },
        "quantization_performed": False,
        "cloud_compute_used": False,
        "sealed_audit_accessed": False,
    }
    if device.type == "cuda":
        summary["gpu"] = torch.cuda.get_device_name(device)
        summary["torch_peak_allocated_vram_bytes"] = max(
            int(results[key]["peak_allocated_vram_bytes"]) for key in CONDITIONS
        )
    else:
        summary["gpu"] = None
        summary["torch_peak_allocated_vram_bytes"] = 0
    (output / "experiment-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-features", type=Path, required=True)
    parser.add_argument("--video-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.audio_features, args.video_features, args.output), indent=2, sort_keys=True
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
