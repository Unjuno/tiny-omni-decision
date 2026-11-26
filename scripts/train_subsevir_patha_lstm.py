"""Fit predeclared frozen-DINOv2 temporal Decision readouts for sub-SEVIR."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn

SEED = 17
BATCH_SIZE = 128
MAX_EPOCHS = 60
PATIENCE = 10
MIN_DELTA = 1e-4
HIDDEN_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
BOOTSTRAP_REPLICATES = 10_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TemporalDecision(nn.Module):
    """One or two frozen-feature LSTMs followed by a small binary scorer."""

    def __init__(self, mode: str) -> None:
        super().__init__()
        if mode not in {"satellite", "radar", "joint"}:
            raise ValueError(f"unsupported readout mode: {mode}")
        self.mode = mode
        if mode in {"satellite", "joint"}:
            self.satellite_lstm = nn.LSTM(768, HIDDEN_SIZE, batch_first=True)
        if mode in {"radar", "joint"}:
            self.radar_lstm = nn.LSTM(768, HIDDEN_SIZE, batch_first=True)
        scorer_in = HIDDEN_SIZE * (2 if mode == "joint" else 1)
        self.scorer = nn.Sequential(
            nn.Linear(scorer_in, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4 or values.shape[1:] != (8, 2, 768):
            raise ValueError(f"expected [batch,8,2,768], got {tuple(values.shape)}")
        pooled = []
        if self.mode in {"satellite", "joint"}:
            _, (hidden, _) = self.satellite_lstm(values[:, :, 0, :])
            pooled.append(hidden[-1])
        if self.mode in {"radar", "joint"}:
            _, (hidden, _) = self.radar_lstm(values[:, :, 1, :])
            pooled.append(hidden[-1])
        return self.scorer(torch.cat(pooled, dim=-1))


def metrics(targets: np.ndarray, logits: np.ndarray) -> dict[str, object]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == targets
    nll_each = -np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1.0))
    one_hot = np.eye(2, dtype=np.float64)[targets]
    brier_each = np.square(probabilities - one_hot).sum(axis=1)
    ece = 0.0
    bins = []
    for bin_index in range(15):
        lower, upper = bin_index / 15.0, (bin_index + 1) / 15.0
        selected = (confidence >= lower) & (
            (confidence < upper) if bin_index < 14 else (confidence <= upper)
        )
        if not selected.any():
            bins.append({"lower": lower, "upper": upper, "count": 0})
            continue
        accuracy = float(correct[selected].mean())
        mean_confidence = float(confidence[selected].mean())
        ece += float(selected.mean()) * abs(accuracy - mean_confidence)
        bins.append({
            "lower": lower,
            "upper": upper,
            "count": int(selected.sum()),
            "accuracy": accuracy,
            "mean_confidence": mean_confidence,
        })
    return {
        "count": int(len(targets)),
        "accuracy": float(correct.mean()),
        "nll": float(nll_each.mean()),
        "brier": float(brier_each.mean()),
        "ece_15": float(ece),
        "mean_confidence": float(confidence.mean()),
        "correct_count": int(correct.sum()),
        "confusion": {
            "tn": int(((targets == 0) & (predictions == 0)).sum()),
            "fp": int(((targets == 0) & (predictions == 1)).sum()),
            "fn": int(((targets == 1) & (predictions == 0)).sum()),
            "tp": int(((targets == 1) & (predictions == 1)).sum()),
        },
        "reliability_bins": bins,
        "probabilities": probabilities,
        "predictions": predictions,
    }


def predict(model: nn.Module, inputs: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(inputs), BATCH_SIZE):
            batch = inputs[start : start + BATCH_SIZE].to(device)
            parts.append(model(batch).float().cpu().numpy())
    return np.concatenate(parts, axis=0)


def _save_state(path: Path, model: nn.Module) -> None:
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()},
        str(path),
    )


def _candidate_fit(
    mode: str,
    train_x: torch.Tensor,
    train_y: np.ndarray,
    val_x: torch.Tensor,
    val_y: np.ndarray,
    val_windows: list[dict[str, object]],
    output: Path,
    device: torch.device,
) -> dict[str, object]:
    candidate_dir = output / mode
    candidate_dir.mkdir(parents=False, exist_ok=False)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model = TemporalDecision(mode).to(device)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    best_nll = math.inf
    best_epoch = 0
    stale_epochs = 0
    history_path = candidate_dir / "training-history.jsonl"
    fit_started = time.perf_counter()

    with history_path.open("w", encoding="utf-8", newline="\n") as history_file:
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            order = torch.randperm(len(train_x), generator=generator)
            train_loss_sum = 0.0
            train_correct = 0
            train_count = 0
            for start in range(0, len(order), BATCH_SIZE):
                batch_ids = order[start : start + BATCH_SIZE]
                batch_x = train_x.index_select(0, batch_ids).to(device)
                batch_y = torch.from_numpy(train_y[batch_ids.numpy()].astype(np.int64)).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x)
                loss = nn.functional.cross_entropy(logits, batch_y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite {mode} loss at epoch {epoch}")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(f"non-finite {mode} gradient at epoch {epoch}")
                optimizer.step()
                train_loss_sum += float(loss.detach()) * len(batch_y)
                train_correct += int((logits.detach().argmax(dim=1) == batch_y).sum())
                train_count += len(batch_y)

            val_logits = predict(model, val_x, device)
            val_metrics = metrics(val_y, val_logits)
            row = {
                "epoch": epoch,
                "train_cross_entropy_rolling": train_loss_sum / train_count,
                "train_accuracy_rolling": train_correct / train_count,
                "validation": {key: val_metrics[key] for key in (
                    "accuracy", "nll", "brier", "ece_15", "mean_confidence"
                )},
                "validation_count": len(val_y),
                "best_epoch_before_update": best_epoch,
                "selector": "minimum validation NLL, min_delta=1e-4",
                "gradient_clip_norm": GRADIENT_CLIP,
                "elapsed_seconds": time.perf_counter() - fit_started,
            }
            history_file.write(json.dumps(row) + "\n")
            history_file.flush()
            print(json.dumps({"event": "epoch", "candidate": mode, **row}), flush=True)

            current_nll = float(val_metrics["nll"])
            if current_nll < best_nll - MIN_DELTA:
                best_nll = current_nll
                best_epoch = epoch
                stale_epochs = 0
                _save_state(candidate_dir / "best.safetensors", model)
            else:
                stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break

    _save_state(candidate_dir / "final.safetensors", model)

    selected_model = TemporalDecision(mode).to(device)
    selected_model.load_state_dict(load_file(str(candidate_dir / "best.safetensors"), device="cpu"))
    selected_model.eval()
    reloaded_logits = predict(selected_model, val_x, device)
    selected_metrics = metrics(val_y, reloaded_logits)
    selected_predictions = selected_metrics["predictions"]
    selected_probabilities = selected_metrics["probabilities"]
    predictions_path = candidate_dir / "validation-predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as prediction_file:
        for window, target, prediction, probabilities in zip(
            val_windows, val_y, selected_predictions, selected_probabilities, strict=True
        ):
            prediction_file.write(json.dumps({
                "sample_id": window["sample_id"],
                "event_id": window["event_id"],
                "target": int(target),
                "prediction": int(prediction),
                "probabilities": probabilities.tolist(),
                "options": ["no_lightning", "lightning"],
            }) + "\n")
    summary = {
        "mode": mode,
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "best_validation_nll": best_nll,
        "selected_validation_metrics": {
            key: selected_metrics[key] for key in (
                "count", "accuracy", "nll", "brier", "ece_15", "mean_confidence",
                "correct_count", "confusion", "reliability_bins",
            )
        },
        "trainable_readout_parameters": trainable_parameters,
        "feature_encoder_parameters_frozen": 22_056_576,
        "total_candidate_parameters": 22_056_576 + trainable_parameters,
        "fit_wall_seconds": time.perf_counter() - fit_started,
        "best_checkpoint_sha256": sha256_file(candidate_dir / "best.safetensors"),
        "final_checkpoint_sha256": sha256_file(candidate_dir / "final.safetensors"),
        "prediction_file_sha256": sha256_file(predictions_path),
    }
    (candidate_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    summary["_logits"] = reloaded_logits
    summary["_probabilities"] = selected_probabilities
    summary["_predictions"] = selected_predictions
    return summary


def _group_bootstrap(
    targets: np.ndarray,
    outcomes_a: np.ndarray,
    nll_a: np.ndarray,
    brier_a: np.ndarray,
    group_ids: list[str],
    outcomes_b: np.ndarray | None = None,
    nll_b: np.ndarray | None = None,
    brier_b: np.ndarray | None = None,
) -> dict[str, object]:
    unique_groups = sorted(set(group_ids))
    group_positions = {group: index for index, group in enumerate(unique_groups)}
    sums_a = np.zeros((len(unique_groups), 4), dtype=np.float64)
    sums_b = np.zeros_like(sums_a) if outcomes_b is not None else None
    counts = np.zeros(len(unique_groups), dtype=np.float64)
    for row, group in enumerate(group_ids):
        index = group_positions[group]
        sums_a[index, 0] += float(outcomes_a[row])
        sums_a[index, 1] += float(nll_a[row])
        sums_a[index, 2] += float(brier_a[row])
        counts[index] += 1.0
        if sums_b is not None:
            assert nll_b is not None and brier_b is not None and outcomes_b is not None
            sums_b[index, 0] += float(outcomes_b[row])
            sums_b[index, 1] += float(nll_b[row])
            sums_b[index, 2] += float(brier_b[row])
    rng = np.random.default_rng(SEED)
    sampled = rng.integers(0, len(unique_groups), size=(BOOTSTRAP_REPLICATES, len(unique_groups)))
    denominator = counts[sampled].sum(axis=1)
    result: dict[str, object] = {"group_count": len(unique_groups)}
    for index, name in enumerate(("accuracy", "nll", "brier")):
        values = sums_a[:, index][sampled].sum(axis=1) / denominator
        result[name] = [float(value) for value in np.quantile(values, [0.025, 0.975])]
    if sums_b is not None:
        diffs = sums_a - sums_b
        result["paired_difference_a_minus_b"] = {
            name: [
                float(value)
                for value in np.quantile(
                    diffs[:, index][sampled].sum(axis=1) / denominator,
                    [0.025, 0.975],
                )
            ]
            for index, name in enumerate(("accuracy", "nll", "brier"))
        }
    return result


def _per_example_losses(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    nll = -np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1.0))
    one_hot = np.eye(2, dtype=np.float64)[targets]
    brier = np.square(probabilities - one_hot).sum(axis=1)
    return nll, brier


def _serialize_candidate(summary: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in summary.items() if not key.startswith("_")}


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite training artifact: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest_hash = sha256_file(args.manifest)
    cache_metadata = json.loads((args.feature_cache / "metadata.json").read_text())
    if cache_metadata["manifest_sha256"] != manifest_hash:
        raise ValueError("feature cache belongs to a different frozen manifest")
    if cache_metadata["model_revision"] != "ed25f3a31f01632728cabb09d1542f84ab7b0056":
        raise ValueError("feature cache has an unexpected encoder revision")
    features_path = args.feature_cache / "frame_features_fp32.npy"
    if sha256_file(features_path) != cache_metadata["feature_file_sha256"]:
        raise ValueError("feature cache hash mismatch")

    indices = np.load(args.feature_cache / "frame_indices.npy", allow_pickle=False)
    splits = np.load(args.feature_cache / "frame_splits.npy", allow_pickle=False)
    event_ids = np.load(args.feature_cache / "frame_event_ids.npy", allow_pickle=False)
    features = np.load(features_path, mmap_mode="r", allow_pickle=False)
    if features.shape != (len(indices), 2, 768) or not np.isfinite(features).all():
        raise ValueError("feature cache shape or finite-value check failed")
    frame_positions = {int(source_index): position for position, source_index in enumerate(indices)}
    if len(frame_positions) != len(indices):
        raise ValueError("feature cache has duplicate frame indices")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    windows = manifest["windows"]
    grouped_event_ids = manifest["split_group_ids"]
    if set(grouped_event_ids["train"]) & set(grouped_event_ids["validation"]):
        raise ValueError("frozen manifest has event overlap")

    rows: dict[str, list[dict[str, object]]] = {"train": [], "validation": []}
    x_values: dict[str, list[np.ndarray]] = {"train": [], "validation": []}
    for window in windows:
        split = window["split"]
        group_set = set(grouped_event_ids[split])
        if window["event_id"] not in group_set:
            raise ValueError(f"window event not in its frozen split: {window['sample_id']}")
        sequence_features = []
        for source_index in window["input_indices"]:
            if source_index not in frame_positions:
                raise ValueError(f"missing cached input frame: {source_index}")
            position = frame_positions[source_index]
            if splits[position] != split or event_ids[position] != window["event_id"]:
                raise ValueError(f"cached frame split/identity mismatch: {window['sample_id']}")
            sequence_features.append(np.asarray(features[position], dtype=np.float32))
        x_values[split].append(np.stack(sequence_features))
        rows[split].append(window)
    x_arrays = {
        split: np.stack(values).astype(np.float32, copy=False)
        for split, values in x_values.items()
    }
    y_arrays = {
        split: np.asarray([int(row["target"]) for row in rows[split]], dtype=np.int64)
        for split in rows
    }
    if len(y_arrays["train"]) != 15_828 or len(y_arrays["validation"]) != 3_988:
        raise ValueError("training/validation window counts differ from frozen manifest")
    train_x = torch.from_numpy(x_arrays["train"])
    val_x = torch.from_numpy(x_arrays["validation"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for readout fitting")
    device = torch.device("cuda")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats(device)
    run_started = time.perf_counter()
    candidates = {}
    for mode in ("satellite", "radar", "joint"):
        candidates[mode] = _candidate_fit(
            mode,
            train_x,
            y_arrays["train"],
            val_x,
            y_arrays["validation"],
            rows["validation"],
            args.output_dir,
            device,
        )

    targets = y_arrays["validation"]
    group_ids = [str(window["event_id"]) for window in rows["validation"]]
    class_rate = float(y_arrays["train"].mean())
    majority_class = int(class_rate >= 0.5)
    baseline_probabilities = np.tile(np.asarray([1.0 - class_rate, class_rate]), (len(targets), 1))
    uniform_probabilities = np.full((len(targets), 2), 0.5)
    baseline_predictions = np.full(len(targets), majority_class)
    for _mode, result in candidates.items():
        probabilities = result["_probabilities"]
        predictions = result["_predictions"]
        nll, brier = _per_example_losses(targets, probabilities)
        correct = (predictions == targets).astype(np.float64)
        ci = _group_bootstrap(targets, correct, nll, brier, group_ids)
        result["selected_validation_metrics"]["event_group_bootstrap_95pct_ci"] = ci

    uniform_metrics = metrics(targets, np.log(uniform_probabilities))
    baseline_metrics = metrics(targets, np.log(baseline_probabilities))
    baseline = {
        "train_positive_rate": class_rate,
        "majority_class": majority_class,
        "majority_accuracy": float((baseline_predictions == targets).mean()),
        "uniform": {key: uniform_metrics[key] for key in (
            "accuracy", "nll", "brier", "ece_15", "mean_confidence"
        )},
        "train_prevalence_probability": {key: baseline_metrics[key] for key in (
            "accuracy", "nll", "brier", "ece_15", "mean_confidence"
        )},
    }
    paired = {}
    for other in ("satellite", "radar"):
        joint = candidates["joint"]
        comparator = candidates[other]
        joint_probs = joint["_probabilities"]
        other_probs = comparator["_probabilities"]
        joint_pred = joint["_predictions"]
        other_pred = comparator["_predictions"]
        joint_nll, joint_brier = _per_example_losses(targets, joint_probs)
        other_nll, other_brier = _per_example_losses(targets, other_probs)
        paired["joint_minus_" + other] = _group_bootstrap(
            targets,
            (joint_pred == targets).astype(np.float64),
            joint_nll,
            joint_brier,
            group_ids,
            (other_pred == targets).astype(np.float64),
            other_nll,
            other_brier,
        )

    joint_model = TemporalDecision("joint").to(device)
    joint_model.load_state_dict(
        load_file(str(args.output_dir / "joint" / "best.safetensors"), device="cpu")
    )
    control_inputs = x_arrays["validation"].copy()
    control_results = {}
    for name, modified in (
        ("intact", control_inputs),
        ("zero_satellite", np.concatenate((
            np.zeros_like(control_inputs[:, :, 0:1, :]), control_inputs[:, :, 1:2, :]
        ), axis=2)),
        ("zero_radar", np.concatenate((
            control_inputs[:, :, 0:1, :], np.zeros_like(control_inputs[:, :, 1:2, :])
        ), axis=2)),
        ("time_reversed", np.flip(control_inputs, axis=1).copy()),
    ):
        logits = predict(joint_model, torch.from_numpy(modified), device)
        control_metrics = metrics(targets, logits)
        probabilities = control_metrics["probabilities"]
        control_results[name] = {key: control_metrics[key] for key in (
            "accuracy", "nll", "brier", "ece_15", "mean_confidence", "correct_count"
        )}
        if name != "intact":
            nll, brier = _per_example_losses(targets, probabilities)
            paired[name + "_minus_intact"] = _group_bootstrap(
                targets,
                (control_metrics["predictions"] == targets).astype(np.float64),
                nll,
                brier,
                group_ids,
                (candidates["joint"]["_predictions"] == targets).astype(np.float64),
                joint_nll,
                joint_brier,
            )

    total_wall = time.perf_counter() - run_started
    result = {
        "schema": "tiny-omni-decision/subsevir-path-a-lstm-results-v1",
        "status": "complete",
        "manifest_sha256": manifest_hash,
        "feature_metadata_sha256": sha256_file(args.feature_cache / "metadata.json"),
        "feature_cache": cache_metadata,
        "protocol": {
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "min_delta": MIN_DELTA,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRADIENT_CLIP,
            "selector": "minimum fixed-validation NLL",
            "calibration": "none fitted; ECE is reported from raw softmax probabilities",
        },
        "task": (
            "predict next 5-minute frame lightning presence from preceding "
            "eight 5-minute frames"
        ),
        "options": ["no_lightning", "lightning"],
        "train_window_count": len(y_arrays["train"]),
        "validation_window_count": len(targets),
        "train_event_groups": len(set(str(row["event_id"]) for row in rows["train"])),
        "validation_event_groups": len(set(group_ids)),
        "training_loss_note": (
            "rolling optimizer-fit metrics are not fixed-example "
            "generalization estimates"
        ),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device),
        "wall_seconds": total_wall,
        "baseline": baseline,
        "candidates": {mode: _serialize_candidate(value) for mode, value in candidates.items()},
        "paired_group_bootstrap_95pct": paired,
        "joint_input_controls": control_results,
        "bootstrap": {
            "unit": "storm event ID",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": SEED,
            "interval": "percentile 95%",
        },
        "claim_scope": "one project-created sub-SEVIR five-minute lightning forecast task only",
        "sealed_audit_accessed": False,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
