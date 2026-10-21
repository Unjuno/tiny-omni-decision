"""Train a tiny sequence readout over ordered PCH sensor windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from scripts.pch_moment_sensor_probe import (
    ACTIVITY_IDS,
    ARCHIVE_SHA256,
    ARCHIVE_SIZE,
    build_records,
    preflight_inventory,
    sha256_bytes,
    sha256_file,
    subject_split_for_record,
)

CONFIG = {
    "experiment_id": "pch-recording-temporal-attempt-01",
    "seed": 17,
    "window_encoder_hidden_size": 64,
    "window_encoder_source": "pch-lstm-sensor-attempt-04 selected best checkpoint, frozen",
    "temporal_lstm_hidden_size": 32,
    "temporal_lstm_layers": 1,
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "record_batch_size": 16,
    "feature_window_batch_size": 256,
    "max_epochs": 100,
    "early_stopping_patience": 10,
    "min_delta": 1e-5,
    "gradient_clip_norm": 1.0,
    "optimizer": "AdamW",
    "checkpoint_selection": "per-arm minimum development-validation recording NLL",
    "conditions": ["chronological", "fixed-per-record shuffle control"],
    "shuffled_order": "SHA-256(seed:sample_id) then NumPy PCG64 permutation",
    "validation_policy": "development only; fixed participant split and IDs from attempt-04",
    "heldout_policy": "do not load heldout labels or signals in this attempt",
}
BASELINE_HASHES = {
    "preflight.json": "963ad2c1517e04a40015b0b32c7a36dae9ab48fc5de3797e325667939af5136f",
    "lstm_best.pt": "b98cd679280db514a091076bbc7a688ce1e1512c912e9553a114a7ead168facd",
    "result.json": "829469373588807fea22ada0afb019fc5895b264ca5845f5227f3e57320b8606",
    "validation-predictions.jsonl": (
        "0d6128df7ba3ca8931460927d5c574891917f801c087bf3310c65631eef622fa"
    ),
}


def paired_subject_bootstrap(
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
    targets: np.ndarray,
    subject_ids: list[str],
    *,
    seed: int = 17,
    replicates: int = 5000,
) -> dict[str, object]:
    """Estimate paired metric differences by resampling whole participants."""
    left = np.asarray(probabilities_a, dtype=np.float64)
    right = np.asarray(probabilities_b, dtype=np.float64)
    target = np.asarray(targets, dtype=np.int64)
    if (
        left.ndim != 2
        or right.shape != left.shape
        or len(target) != len(left)
        or len(subject_ids) != len(target)
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or not np.allclose(left.sum(axis=1), 1.0, atol=1e-6)
        or not np.allclose(right.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise ValueError("bootstrap inputs must contain aligned normalized probabilities")
    if replicates < 100 or len(set(subject_ids)) < 2:
        raise ValueError("bootstrap requires at least 100 replicates and two subjects")
    if np.any(target < 0) or np.any(target >= left.shape[1]):
        raise ValueError("target class is outside the probability columns")

    subjects = np.asarray(sorted(set(subject_ids)))
    owner = np.asarray(subject_ids)
    rng = np.random.default_rng(seed)
    deltas: dict[str, list[float]] = {
        "accuracy": [],
        "nll": [],
        "brier": [],
    }
    for _ in range(replicates):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        indices = np.concatenate([np.flatnonzero(owner == subject) for subject in sampled])
        labels = target[indices]
        for name, values in (("a", left[indices]), ("b", right[indices])):
            predictions = values.argmax(axis=1)
            if name == "a":
                correct_a = float(np.mean(predictions == labels))
                nll_a = float(
                    -np.log(np.clip(values[np.arange(len(labels)), labels], 1e-12, 1.0)).mean()
                )
                brier_a = float(
                    np.square(values - np.eye(left.shape[1])[labels]).sum(axis=1).mean()
                )
            else:
                deltas["accuracy"].append(correct_a - float(np.mean(predictions == labels)))
                deltas["nll"].append(
                    nll_a
                    - float(
                        -np.log(np.clip(values[np.arange(len(labels)), labels], 1e-12, 1.0)).mean()
                    )
                )
                deltas["brier"].append(
                    brier_a
                    - float(np.square(values - np.eye(left.shape[1])[labels]).sum(axis=1).mean())
                )

    def interval(values: list[float]) -> list[float]:
        return [float(value) for value in np.quantile(values, [0.025, 0.975])]

    return {
        "method": "paired participant-cluster bootstrap, percentile 95% interval",
        "seed": seed,
        "replicates": replicates,
        "participant_count": int(len(subjects)),
        "delta_direction": "condition_a_minus_condition_b",
        "accuracy_difference_95_percent_interval": interval(deltas["accuracy"]),
        "nll_difference_95_percent_interval": interval(deltas["nll"]),
        "brier_difference_95_percent_interval": interval(deltas["brier"]),
    }


def build_window_encoder(torch, hidden_size: int):
    """Return the LSTM representation layer used by the frozen window baseline."""
    from torch import nn

    class WindowEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(3, hidden_size, batch_first=True)

        def forward(self, windows):
            _, (hidden, _) = self.lstm(windows)
            return hidden[-1]

    return WindowEncoder()


def split_window_embeddings(embeddings: Any, lengths: list[int]) -> list[np.ndarray]:
    """Split flat window embeddings without changing their within-record order."""
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("window embeddings must be a 2D array")
    if any(length <= 0 for length in lengths):
        raise ValueError("every recording must contain a positive number of windows")
    if sum(lengths) != len(values):
        raise ValueError("window length sum must equal the embedding count")
    offsets = np.cumsum([0, *lengths])
    return [values[offsets[i] : offsets[i + 1]] for i in range(len(lengths))]


def pad_sequence_batch(sequences: list[Any]) -> tuple[np.ndarray, list[int]]:
    """Zero-pad the sequence tails and return true lengths for packed recurrence."""
    arrays = [np.asarray(sequence, dtype=np.float32) for sequence in sequences]
    if not arrays or any(array.ndim != 2 or len(array) == 0 for array in arrays):
        raise ValueError("batch cannot contain an empty or malformed sequence")
    width = arrays[0].shape[1]
    if any(array.shape[1] != width for array in arrays):
        raise ValueError("all sequences must share the same feature width")
    lengths = [len(array) for array in arrays]
    padded = np.zeros((len(arrays), max(lengths), width), dtype=np.float32)
    for index, array in enumerate(arrays):
        padded[index, : len(array)] = array
    return padded, lengths


def build_temporal_classifier(torch, input_size: int, hidden_size: int, class_count: int):
    """Build a sequence-level LSTM whose packed input ignores batch padding."""
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence

    class TemporalClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.temporal_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.classifier = nn.Linear(hidden_size, class_count)

        def forward(self, padded, lengths):
            packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
            _, (hidden, _) = self.temporal_lstm(packed)
            return self.classifier(hidden[-1])

    return TemporalClassifier()


def sequence_order_indices(
    length: int, *, mode: str, seed: int = 17, sample_id: str = ""
) -> list[int]:
    """Return a deterministic within-recording order for a window sequence."""
    if length <= 0:
        raise ValueError("sequence length must be positive")
    if mode == "chronological":
        return list(range(length))
    if mode == "reverse":
        return list(range(length - 1, -1, -1))
    if mode == "shuffle":
        seed_bytes = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()[:8]
        local_seed = int.from_bytes(seed_bytes, byteorder="little", signed=False)
        return np.random.default_rng(local_seed).permutation(length).tolist()
    raise ValueError(f"unknown sequence order mode: {mode}")


def classification_metrics(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if (
        probabilities.ndim != 2
        or probabilities.shape[0] != len(targets)
        or not np.isfinite(probabilities).all()
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise ValueError("expected finite normalized class probabilities")
    predictions = probabilities.argmax(axis=1)
    correct = predictions == targets
    class_names = (
        list(ACTIVITY_IDS)
        if probabilities.shape[1] == len(ACTIVITY_IDS)
        else [str(index) for index in range(probabilities.shape[1])]
    )
    class_accuracy = {
        class_names[class_id]: float(correct[targets == class_id].mean())
        for class_id in range(probabilities.shape[1])
        if np.any(targets == class_id)
    }
    class_counts = {
        class_names[class_id]: int(np.sum(targets == class_id))
        for class_id in range(probabilities.shape[1])
    }
    confidence = probabilities.max(axis=1)
    ece = 0.0
    for index in range(15):
        low, high = index / 15, (index + 1) / 15
        mask = (confidence >= low) & (confidence < high if index < 14 else confidence <= high)
        if mask.any():
            ece += float(mask.mean()) * abs(float(confidence[mask].mean() - correct[mask].mean()))
    one_hot = np.eye(probabilities.shape[1])[targets]
    return {
        "n_recordings": len(targets),
        "accuracy": float(correct.mean()),
        "macro_accuracy_present_classes": float(np.mean(list(class_accuracy.values()))),
        "class_counts": class_counts,
        "class_accuracy": class_accuracy,
        "nll": float(
            -np.log(
                np.clip(
                    probabilities[np.arange(len(targets)), targets],
                    np.finfo(np.float64).tiny,
                    1.0,
                )
            ).mean()
        ),
        "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ece_15": ece,
        "mean_confidence": float(confidence.mean()),
    }


def _predict_sequences(torch, model, sequences, *, device, batch_size):
    model.eval()
    rows = []
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            selected = sequences[start : start + batch_size]
            padded, lengths = pad_sequence_batch(selected)
            values = torch.as_tensor(padded, dtype=torch.float32, device=device)
            logits = model(values, lengths)
            if not torch.isfinite(logits).all():
                raise RuntimeError("nonfinite temporal logits")
            rows.append(logits.float().cpu().numpy())
    return np.concatenate(rows, axis=0)


def _atomic_torch_save(torch, path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f"{path.stem}-", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def train_sequence_model(
    torch,
    *,
    train_sequences: list[np.ndarray],
    train_targets: np.ndarray,
    train_sample_ids: list[str],
    validation_sequences: list[np.ndarray],
    validation_targets: np.ndarray,
    validation_sample_ids: list[str],
    order_mode: str,
    output_dir: Path,
    config: dict[str, Any],
    device: str,
    class_count: int,
) -> dict[str, Any]:
    """Fit one small sequence-level LSTM, selecting and reloading by validation NLL."""
    for sequences, targets, sample_ids, split_name in (
        (train_sequences, train_targets, train_sample_ids, "train"),
        (validation_sequences, validation_targets, validation_sample_ids, "validation"),
    ):
        if not sequences or len(sequences) != len(targets) or len(sequences) != len(sample_ids):
            raise ValueError(f"{split_name} sequences, targets and IDs must align")
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError(f"{split_name} sample IDs must be unique")
        if any(np.asarray(sequence).ndim != 2 or len(sequence) == 0 for sequence in sequences):
            raise ValueError(f"{split_name} sequences must be nonempty 2D arrays")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite attempt output: {output_dir}")
    output_dir.mkdir(parents=True)
    train_targets = np.asarray(train_targets, dtype=np.int64)
    validation_targets = np.asarray(validation_targets, dtype=np.int64)
    input_size = int(np.asarray(train_sequences[0]).shape[1])
    if any(
        np.asarray(sequence).shape[1] != input_size
        for sequence in train_sequences + validation_sequences
    ):
        raise ValueError("all temporal sequences must share the same embedding width")
    if np.any(train_targets < 0) or np.any(train_targets >= class_count):
        raise ValueError("training target is outside the fixed class range")
    if np.any(validation_targets < 0) or np.any(validation_targets >= class_count):
        raise ValueError("validation target is outside the fixed class range")

    def ordered(sequences, sample_ids):
        return [
            np.asarray(sequence, dtype=np.float32)[
                sequence_order_indices(
                    len(sequence),
                    mode=order_mode,
                    seed=int(config["seed"]),
                    sample_id=sample_id,
                )
            ]
            for sequence, sample_id in zip(sequences, sample_ids, strict=True)
        ]

    fit_inputs = ordered(train_sequences, train_sample_ids)
    val_inputs = ordered(validation_sequences, validation_sample_ids)
    seed = int(config["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = build_temporal_classifier(
        torch,
        input_size=input_size,
        hidden_size=int(config["hidden_size"]),
        class_count=class_count,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    if device != "cpu":
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    history = []
    best_nll = float("inf")
    best_epoch = 0
    best_probabilities = None
    best_metrics = None
    best_state = None
    start_time = time.perf_counter()
    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        permutation = torch.randperm(len(fit_inputs), device="cpu").tolist()
        weighted_loss = 0.0
        gradient_norm = 0.0
        batch_size = int(config["batch_size"])
        for start in range(0, len(permutation), batch_size):
            selected = permutation[start : start + batch_size]
            padded, lengths = pad_sequence_batch([fit_inputs[index] for index in selected])
            inputs = torch.as_tensor(padded, dtype=torch.float32, device=device)
            targets = torch.as_tensor(train_targets[selected], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs, lengths)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite temporal LSTM loss at epoch {epoch}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["gradient_clip_norm"])
            )
            if not torch.isfinite(norm):
                raise RuntimeError(f"nonfinite temporal LSTM gradient at epoch {epoch}")
            optimizer.step()
            weighted_loss += float(loss.detach().cpu()) * len(selected)
            gradient_norm = float(norm.detach().cpu())

        train_logits = _predict_sequences(
            torch, model, fit_inputs, device=device, batch_size=batch_size
        )
        validation_logits = _predict_sequences(
            torch, model, val_inputs, device=device, batch_size=batch_size
        )
        train_probabilities = torch.as_tensor(train_logits).softmax(dim=-1).numpy()
        validation_probabilities = torch.as_tensor(validation_logits).softmax(dim=-1).numpy()
        train_metrics = classification_metrics(train_probabilities, train_targets)
        validation_metrics = classification_metrics(validation_probabilities, validation_targets)
        history_item = {
            "epoch": epoch,
            "train_ce": weighted_loss / len(fit_inputs),
            "train_accuracy": train_metrics["accuracy"],
            "validation_nll": validation_metrics["nll"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_accuracy": validation_metrics["macro_accuracy_present_classes"],
            "validation_brier": validation_metrics["brier"],
            "validation_ece_15": validation_metrics["ece_15"],
            "last_batch_gradient_norm": gradient_norm,
        }
        history.append(history_item)
        with (output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(history_item, sort_keys=True) + "\n")
        if validation_metrics["nll"] < best_nll - float(config["min_delta"]):
            best_nll = float(validation_metrics["nll"])
            best_epoch = epoch
            best_metrics = validation_metrics
            best_probabilities = validation_probabilities.copy()
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            _atomic_torch_save(
                torch,
                output_dir / "best.pt",
                {"state_dict": best_state, "selected_epoch": best_epoch, "order_mode": order_mode},
            )
        elif epoch - best_epoch >= int(config["patience"]):
            break

    elapsed = time.perf_counter() - start_time
    final_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    _atomic_torch_save(
        torch,
        output_dir / "final.pt",
        {"state_dict": final_state, "final_epoch": history[-1]["epoch"], "order_mode": order_mode},
    )
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    selected = torch.load(output_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(selected["state_dict"])
    reloaded_logits = _predict_sequences(
        torch, model, val_inputs, device=device, batch_size=int(config["batch_size"])
    )
    reloaded_probabilities = torch.as_tensor(reloaded_logits).softmax(dim=-1).numpy()
    if not np.array_equal(best_probabilities, reloaded_probabilities):
        raise RuntimeError("best checkpoint reload did not reproduce validation probabilities")
    if classification_metrics(reloaded_probabilities, validation_targets) != best_metrics:
        raise RuntimeError("best checkpoint reload metrics differ from selected validation metrics")
    return {
        "order_mode": order_mode,
        "best_epoch": best_epoch,
        "stopping_epoch": history[-1]["epoch"],
        "elapsed_seconds": elapsed,
        "epochs": history,
        "metrics": best_metrics,
        "probabilities": reloaded_probabilities,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_checkpoint_reload_exact": True,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(torch.device(device))) if device != "cpu" else 0
        ),
        "peak_cuda_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(torch.device(device))) if device != "cpu" else 0
        ),
    }


def predict_sequence_model(
    torch,
    checkpoint_path: Path,
    sequences: list[np.ndarray],
    sample_ids: list[str],
    *,
    order_mode: str,
    config: dict[str, Any],
    device: str,
    class_count: int,
) -> np.ndarray:
    """Reload one selected checkpoint and score the same rows under a fixed order."""
    if len(sequences) != len(sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sequences and unique sample IDs must align")
    ordered_sequences = [
        np.asarray(sequence, dtype=np.float32)[
            sequence_order_indices(
                len(sequence),
                mode=order_mode,
                seed=int(config["seed"]),
                sample_id=sample_id,
            )
        ]
        for sequence, sample_id in zip(sequences, sample_ids, strict=True)
    ]
    input_size = int(ordered_sequences[0].shape[1])
    model = build_temporal_classifier(
        torch,
        input_size=input_size,
        hidden_size=int(config["hidden_size"]),
        class_count=class_count,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    logits = _predict_sequences(
        torch,
        model,
        ordered_sequences,
        device=device,
        batch_size=int(config["batch_size"]),
    )
    probabilities = torch.as_tensor(logits).softmax(dim=-1).numpy()
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-6
    ):
        raise RuntimeError("replayed sequence probabilities are invalid")
    return probabilities


def _load_training_validation(archive_path: Path):
    with zipfile.ZipFile(archive_path) as archive:
        records, split = build_records(archive)
    fit = [record for record in records if subject_split_for_record(record, split) == "fit"]
    validation = [
        record for record in records if subject_split_for_record(record, split) == "validation"
    ]
    order_text = "\n".join(
        f"{record['sample_id']}:{index}"
        for record in records
        for index in range(record["window_count"])
    )
    return records, split, fit, validation, sha256_bytes(order_text.encode())


def _load_baseline_artifacts(baseline_dir: Path):
    expected = {name: baseline_dir / name for name in BASELINE_HASHES}
    for name, path in expected.items():
        if not path.is_file() or sha256_file(path) != BASELINE_HASHES[name]:
            raise ValueError(f"frozen PCH window-LSTM baseline hash mismatch: {name}")
    old_preflight = json.loads((baseline_dir / "preflight.json").read_text(encoding="utf-8"))
    baseline_result = json.loads((baseline_dir / "result.json").read_text(encoding="utf-8"))
    prediction_rows = [
        json.loads(line)
        for line in (baseline_dir / "validation-predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    return old_preflight, baseline_result, prediction_rows


def prepare_preflight(archive_path: Path, baseline_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Freeze the source, exact old split, validation IDs and comparison conditions."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite attempt output: {output_dir}")
    if archive_path.stat().st_size != ARCHIVE_SIZE or sha256_file(archive_path) != ARCHIVE_SHA256:
        raise ValueError("PCH archive size/SHA-256 differs from the acquired official source")
    old_preflight, baseline_result, baseline_rows = _load_baseline_artifacts(baseline_dir)
    records, split, fit, validation, window_order_sha256 = _load_training_validation(archive_path)
    inventory = preflight_inventory(records, split, batch_size=CONFIG["feature_window_batch_size"])
    if window_order_sha256 != old_preflight["window_order_sha256"]:
        raise ValueError("ordered fit/validation windows differ from the frozen baseline")
    if (
        inventory["fit_sample_id_order_sha256"]
        != old_preflight["inventory"]["fit_sample_id_order_sha256"]
    ):
        raise ValueError("fit recording order differs from the frozen baseline")
    expected_validation_hash = baseline_result["validation_id_order_sha256"]
    if inventory["validation_sample_id_order_sha256"] != expected_validation_hash:
        raise ValueError("validation recording order differs from the frozen baseline")
    if split != old_preflight["split_metadata"]:
        raise ValueError("participant split differs from the frozen baseline")
    if len(baseline_rows) != len(validation):
        raise ValueError("baseline predictions do not match validation recording count")
    validation_ids = [record["sample_id"] for record in validation]
    if [row.get("sample_id") for row in baseline_rows] != validation_ids:
        raise ValueError("baseline prediction IDs do not match the pinned validation order")
    if any(
        row.get("target_id") != record["label"]
        for row, record in zip(baseline_rows, validation, strict=True)
    ):
        raise ValueError("baseline prediction labels differ from the frozen validation labels")
    if any(
        len(row.get("lstm_probabilities", [])) != len(ACTIVITY_IDS)
        or not np.isfinite(row["lstm_probabilities"]).all()
        for row in baseline_rows
    ):
        raise ValueError("baseline probabilities are missing or invalid")

    script_path = Path(__file__).resolve()
    guard_path = script_path.with_name("run_with_hotgpu_guard.py")
    result = {
        "status": "PASS_PCH_RECORDING_TEMPORAL_PREFLIGHT",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "script_sha256": sha256_file(script_path),
        "existing_hotgpu_guard_sha256": sha256_file(guard_path),
        "archive_path": str(archive_path.resolve()),
        "archive_bytes": ARCHIVE_SIZE,
        "archive_sha256": ARCHIVE_SHA256,
        "baseline_dir": str(baseline_dir.resolve()),
        "baseline_artifact_hashes": BASELINE_HASHES,
        "baseline_selected_epoch": baseline_result["training"]["selected_epoch"],
        "baseline_stopping_epoch": baseline_result["training"]["stopping_epoch"],
        "baseline_trainable_parameters": baseline_result["model"]["trainable_parameters"],
        "baseline_result_status": baseline_result["status"],
        "baseline_preflight_sha256": sha256_file(baseline_dir / "preflight.json"),
        "split_metadata": split,
        "inventory": inventory,
        "window_order_sha256": window_order_sha256,
        "fit_unique_examples": len(fit),
        "validation_unique_examples": len(validation),
        "fit_unique_participants": len({record["subject_id"] for record in fit}),
        "validation_unique_participants": len({record["subject_id"] for record in validation}),
        "fit_unique_windows": sum(record["window_count"] for record in fit),
        "validation_unique_windows": sum(record["window_count"] for record in validation),
        "baseline_validation_sample_id_order_sha256": expected_validation_hash,
        "validation_sample_ids": validation_ids,
        "conditions": CONFIG["conditions"],
        "effective_config": CONFIG,
        "heldout_payload_opened": False,
        "sealed_audit_loaded": False,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "preflight.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "effective-config.json").write_text(
        json.dumps(CONFIG, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "command.txt").write_text(
        "python -m scripts.run_with_hotgpu_guard --max-wait-seconds 3600 "
        "--reserve-mib 5120 --block-if-free-mib 14000 --min-child-free-mib 8000 -- "
        "<Python 3.11> -m scripts.pch_recording_temporal_probe run "
        f"--archive {archive_path.resolve()} --baseline-dir {baseline_dir.resolve()} "
        f"--output-dir {output_dir.resolve()}\n",
        encoding="utf-8",
    )
    return result


def _encode_windows(torch, records, checkpoint_path: Path, *, device: str, batch_size: int):
    if sha256_file(checkpoint_path) != BASELINE_HASHES["lstm_best.pt"]:
        raise ValueError("frozen window encoder checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    hidden_size = int(state["lstm.weight_hh_l0"].shape[1])
    encoder = build_window_encoder(torch, hidden_size=hidden_size).to(device)
    encoder_state = {
        key.removeprefix("lstm."): value for key, value in state.items() if key.startswith("lstm.")
    }
    encoder.load_state_dict(encoder_state, strict=True)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    mean = checkpoint["axis_mean"].to(device)
    scale = checkpoint["axis_scale"].to(device)
    flat_windows = np.concatenate([record["windows"] for record in records], axis=0)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(flat_windows), batch_size):
            batch = torch.as_tensor(
                flat_windows[start : start + batch_size], dtype=torch.float32, device=device
            )
            normalized = ((batch - mean) / scale).transpose(1, 2).contiguous()
            encoded = encoder(normalized)
            if not torch.isfinite(encoded).all():
                raise RuntimeError("nonfinite frozen window embeddings")
            chunks.append(encoded.float().cpu().numpy())
    flat_features = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    lengths = [int(record["window_count"]) for record in records]
    sequences = split_window_embeddings(flat_features, lengths)
    return encoder, flat_features, sequences


def _metrics_and_prediction_rows(records, probabilities, targets):
    metrics = classification_metrics(probabilities, targets)
    rows = []
    for index, record in enumerate(records):
        rows.append(
            {
                "sample_id": record["sample_id"],
                "subject_id": record["subject_id"],
                "target_id": record["label"],
                "target_index": int(targets[index]),
                "prediction_index": int(np.argmax(probabilities[index])),
                "probabilities": probabilities[index].astype(float).tolist(),
            }
        )
    return metrics, rows


def _artifact_hashes(output_dir: Path) -> dict[str, Any]:
    files = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "artifact-hashes.json":
            files.append(
                {
                    "path": path.relative_to(output_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {"schema_version": 1, "files": files}


def run_experiment(archive_path: Path, baseline_dir: Path, output_dir: Path) -> dict[str, Any]:
    expected_names = {"preflight.json", "effective-config.json", "command.txt"}
    if not output_dir.is_dir() or {path.name for path in output_dir.iterdir()} != expected_names:
        raise ValueError("attempt output must contain only its frozen preflight files")
    preflight_path = output_dir / "preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("script_sha256") != sha256_file(Path(__file__).resolve()):
        raise ValueError("preflight does not match this exact experiment script")
    old_preflight, baseline_result, baseline_rows = _load_baseline_artifacts(baseline_dir)
    if preflight.get("baseline_artifact_hashes") != BASELINE_HASHES:
        raise ValueError("baseline pins differ from the immutable preflight")
    if str(baseline_dir.resolve()) != preflight.get("baseline_dir"):
        raise ValueError("baseline directory differs from the immutable preflight")
    if json.loads((output_dir / "effective-config.json").read_text(encoding="utf-8")) != CONFIG:
        raise ValueError("effective config differs from the pinned experiment config")
    guard_path = Path(__file__).resolve().with_name("run_with_hotgpu_guard.py")
    if sha256_file(guard_path) != preflight.get("existing_hotgpu_guard_sha256"):
        raise ValueError("HotUGPU guard changed after preflight")
    if archive_path.stat().st_size != ARCHIVE_SIZE or sha256_file(archive_path) != ARCHIVE_SHA256:
        raise ValueError("PCH archive size/SHA-256 differs from preflight")
    records, split, fit, validation, order_hash = _load_training_validation(archive_path)
    inventory = preflight_inventory(records, split, batch_size=CONFIG["feature_window_batch_size"])
    if split != preflight["split_metadata"] or inventory != preflight["inventory"]:
        raise ValueError("split or unique-example inventory changed since preflight")
    if order_hash != preflight["window_order_sha256"]:
        raise ValueError("fit/validation window order changed since preflight")
    if sha256_file(baseline_dir / "preflight.json") != preflight["baseline_preflight_sha256"]:
        raise ValueError("original baseline preflight changed since candidate preflight")
    validation_ids = [record["sample_id"] for record in validation]
    if [row.get("sample_id") for row in baseline_rows] != validation_ids:
        raise ValueError("validation IDs no longer match the saved baseline predictions")

    from scripts.pch_moment_sensor_probe import gpu_preflight

    hotgpu = gpu_preflight(dict(os.environ))
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("this training run requires the guarded local CUDA device")
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    start_utc = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(torch.device(device))
    checkpoint_path = baseline_dir / "lstm_best.pt"
    encoder, flat_features, sequences = _encode_windows(
        torch,
        records,
        checkpoint_path,
        device=device,
        batch_size=int(CONFIG["feature_window_batch_size"]),
    )
    feature_extraction_seconds = time.perf_counter() - started
    extraction_peak_allocated = int(torch.cuda.max_memory_allocated(torch.device(device)))
    extraction_peak_reserved = int(torch.cuda.max_memory_reserved(torch.device(device)))
    feature_sha256 = sha256_bytes(flat_features.tobytes(order="C"))
    features_by_id = {
        record["sample_id"]: seq for record, seq in zip(records, sequences, strict=True)
    }
    fit_sequences = [features_by_id[record["sample_id"]] for record in fit]
    validation_sequences = [features_by_id[record["sample_id"]] for record in validation]
    fit_ids = [record["sample_id"] for record in fit]
    val_ids = [record["sample_id"] for record in validation]
    fit_targets = np.asarray(
        [ACTIVITY_IDS.index(record["label"]) for record in fit], dtype=np.int64
    )
    val_targets = np.asarray(
        [ACTIVITY_IDS.index(record["label"]) for record in validation], dtype=np.int64
    )
    validation_subjects = [record["subject_id"] for record in validation]
    baseline_probabilities = np.asarray(
        [row["lstm_probabilities"] for row in baseline_rows], dtype=np.float64
    )
    baseline_metrics, baseline_prediction_rows = _metrics_and_prediction_rows(
        validation, baseline_probabilities, val_targets
    )
    recorded_baseline_metrics = baseline_result["validation"]["metrics"]
    for metric in ("accuracy", "nll", "brier", "ece_15"):
        if not np.isclose(
            baseline_metrics[metric], recorded_baseline_metrics[metric], rtol=1e-10, atol=1e-10
        ):
            raise ValueError(f"saved baseline {metric} does not replay from pinned predictions")

    arm_results = {}
    arm_modes = {"chronological": "chronological", "shuffled_control": "shuffle"}
    training_started = time.perf_counter()
    for arm_name, order_mode in arm_modes.items():
        arm_dir = output_dir / arm_name
        arm_results[arm_name] = train_sequence_model(
            torch,
            train_sequences=fit_sequences,
            train_targets=fit_targets,
            train_sample_ids=fit_ids,
            validation_sequences=validation_sequences,
            validation_targets=val_targets,
            validation_sample_ids=val_ids,
            order_mode=order_mode,
            output_dir=arm_dir,
            config={
                "seed": CONFIG["seed"],
                "hidden_size": CONFIG["temporal_lstm_hidden_size"],
                "batch_size": CONFIG["record_batch_size"],
                "max_epochs": CONFIG["max_epochs"],
                "patience": CONFIG["early_stopping_patience"],
                "min_delta": CONFIG["min_delta"],
                "learning_rate": CONFIG["learning_rate"],
                "weight_decay": CONFIG["weight_decay"],
                "gradient_clip_norm": CONFIG["gradient_clip_norm"],
            },
            device=device,
            class_count=len(ACTIVITY_IDS),
        )
    training_seconds = time.perf_counter() - training_started

    def scores_for_checkpoint(arm_name: str, mode: str):
        return predict_sequence_model(
            torch,
            output_dir / arm_name / "best.pt",
            validation_sequences,
            val_ids,
            order_mode=mode,
            config={
                "seed": CONFIG["seed"],
                "hidden_size": CONFIG["temporal_lstm_hidden_size"],
                "batch_size": CONFIG["record_batch_size"],
            },
            device=device,
            class_count=len(ACTIVITY_IDS),
        )

    chronological_probabilities = arm_results["chronological"]["probabilities"]
    shuffled_probabilities = arm_results["shuffled_control"]["probabilities"]
    order_controls = {
        "chronological_model_reverse": scores_for_checkpoint("chronological", "reverse"),
        "chronological_model_shuffle": scores_for_checkpoint("chronological", "shuffle"),
        "shuffle_model_chronological": scores_for_checkpoint("shuffled_control", "chronological"),
        "shuffle_model_reverse": scores_for_checkpoint("shuffled_control", "reverse"),
    }
    validation_metrics = {
        "window_lstm_baseline": baseline_metrics,
        "chronological": classification_metrics(chronological_probabilities, val_targets),
        "fixed_shuffled_control": classification_metrics(shuffled_probabilities, val_targets),
        **{
            name: classification_metrics(probabilities, val_targets)
            for name, probabilities in order_controls.items()
        },
    }
    bootstrap_order = paired_subject_bootstrap(
        chronological_probabilities,
        shuffled_probabilities,
        val_targets,
        validation_subjects,
        seed=int(CONFIG["seed"]),
        replicates=5000,
    )
    bootstrap_baseline = paired_subject_bootstrap(
        chronological_probabilities,
        baseline_probabilities,
        val_targets,
        validation_subjects,
        seed=int(CONFIG["seed"]),
        replicates=5000,
    )
    prediction_rows = []
    for index, record in enumerate(validation):
        prediction_rows.append(
            {
                "sample_id": record["sample_id"],
                "subject_id": record["subject_id"],
                "target_id": record["label"],
                "target_index": int(val_targets[index]),
                "baseline_window_lstm": baseline_probabilities[index].tolist(),
                "chronological": chronological_probabilities[index].tolist(),
                "fixed_shuffled_control": shuffled_probabilities[index].tolist(),
                **{
                    name: probabilities[index].tolist()
                    for name, probabilities in order_controls.items()
                },
            }
        )
    (output_dir / "validation-predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )
    all_parameters = sum(parameter.numel() for parameter in encoder.parameters())
    temporal_parameters = int(arm_results["chronological"]["trainable_parameters"])
    if temporal_parameters != int(arm_results["shuffled_control"]["trainable_parameters"]):
        raise RuntimeError("paired temporal arms have different parameter counts")
    completed = datetime.now(UTC).isoformat()
    result = {
        "status": "PASS_PCH_RECORDING_TEMPORAL_DEV_COMPARISON",
        "scope": "single development participant split; no independent quality claim",
        "started_at_utc": start_utc,
        "finished_at_utc": completed,
        "source_commit": preflight["source_commit"],
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "preflight_sha256": sha256_file(preflight_path),
        "archive_sha256": ARCHIVE_SHA256,
        "baseline_artifact_hashes": BASELINE_HASHES,
        "window_order_sha256": preflight["window_order_sha256"],
        "feature_cache_sha256": feature_sha256,
        "feature_cache_rows": int(len(flat_features)),
        "training": {
            "records_per_epoch": len(fit),
            "unique_fit_participants": len({record["subject_id"] for record in fit}),
            "unique_windows_per_epoch": int(sum(record["window_count"] for record in fit)),
            "arm_results": {
                name: {key: value for key, value in result.items() if key != "probabilities"}
                for name, result in arm_results.items()
            },
            "total_optimizer_updates": sum(
                int(np.ceil(len(fit) / CONFIG["record_batch_size"]))
                * len(arm_results[name]["epochs"])
                for name in arm_results
            ),
            "total_records_consumed_across_arms": sum(
                len(fit) * len(arm_results[name]["epochs"]) for name in arm_results
            ),
            "elapsed_seconds": training_seconds,
        },
        "validation": {
            "records": len(validation),
            "unique_participants": len(set(validation_subjects)),
            "unique_windows": int(sum(record["window_count"] for record in validation)),
            "sample_id_order_sha256": preflight["baseline_validation_sample_id_order_sha256"],
            "metrics": validation_metrics,
            "paired_subject_bootstrap_chronological_minus_shuffled": bootstrap_order,
            "paired_subject_bootstrap_chronological_minus_baseline": bootstrap_baseline,
            "selection_warning": (
                "This validation split was already used to select the window-LSTM baseline; "
                "both arms are development-only comparisons."
            ),
        },
        "controls": {
            "within_recording_order_sensitivity": {
                name: {
                    "argmax_change_fraction": float(
                        np.mean(
                            probabilities.argmax(axis=1)
                            != chronological_probabilities.argmax(axis=1)
                        )
                    ),
                    "metrics": validation_metrics[name],
                }
                for name, probabilities in order_controls.items()
            },
            "interpretation": (
                "Order perturbation sensitivity is diagnostic; it is not proof that temporal "
                "information is used correctly."
            ),
        },
        "model": {
            "frozen_window_encoder_parameters": int(all_parameters),
            "trainable_temporal_parameters": temporal_parameters,
            "total_parameters_in_candidate_path": int(all_parameters + temporal_parameters),
            "output_classes": list(ACTIVITY_IDS),
        },
        "compute": {
            "gpu_preflight": hotgpu,
            "gpu_name": torch.cuda.get_device_name(device),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "feature_extraction_seconds": feature_extraction_seconds,
            "total_wall_seconds": time.perf_counter() - started,
            "feature_extraction_peak_allocated_bytes": extraction_peak_allocated,
            "feature_extraction_peak_reserved_bytes": extraction_peak_reserved,
            "training_peak_allocated_bytes": max(
                result["peak_cuda_allocated_bytes"] for result in arm_results.values()
            ),
            "training_peak_reserved_bytes": max(
                result["peak_cuda_reserved_bytes"] for result in arm_results.values()
            ),
            "cloud_cost_usd": 0,
        },
        "split": split,
        "heldout_payload_opened": False,
        "sealed_audit_loaded": False,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run_metadata = {
        "status": result["status"],
        "source_commit": result["source_commit"],
        "script_sha256": result["script_sha256"],
        "preflight_sha256": result["preflight_sha256"],
        "effective_config_sha256": sha256_file(output_dir / "effective-config.json"),
        "result_sha256": sha256_file(output_dir / "result.json"),
        "started_at_utc": start_utc,
        "finished_at_utc": completed,
        "started_command": " ".join([str(value) for value in __import__("sys").argv]),
    }
    (output_dir / "run-metadata.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "artifact-hashes.json").write_text(
        json.dumps(_artifact_hashes(output_dir), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        command = commands.add_parser(name)
        command.add_argument("--archive", type=Path, required=True)
        command.add_argument("--baseline-dir", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_preflight(args.archive, args.baseline_dir, args.output_dir)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "fit_records": result["fit_unique_examples"],
                    "validation_records": result["validation_unique_examples"],
                    "heldout_payload_opened": result["heldout_payload_opened"],
                },
                indent=2,
            )
        )
    else:
        result = run_experiment(args.archive, args.baseline_dir, args.output_dir)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "validation_metrics": result["validation"]["metrics"],
                    "total_wall_seconds": result["compute"]["total_wall_seconds"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
