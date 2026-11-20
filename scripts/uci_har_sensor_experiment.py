"""Predeclared, subject-disjoint UCI HAR sensor Decision experiment.

The fit phase reads only the official train files. The audit phase is a
separate, at-most-once command that reads the public official test split only
after the candidate configs and validation-selected checkpoints are frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import random
import subprocess
import sys
import time
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.nn import functional as F

from tiny_omni_decision.native_feature_decision import (
    EnumOptionField,
    NativeFeatureDecisionScorer,
    TypedOptionSchema,
)
from tiny_omni_decision.sensor_temporal_stem import SensorTemporalDecisionScorer

CHANNELS = (
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
)
ACTIVITIES = (
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
)
SEED = 17
VALIDATION_SUBJECTS = 4
EPOCHS = 12
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
HIDDEN_SIZE = 64
READOUT_HIDDEN_SIZE = 128
TIME_UNIT_MS = 1000.0
WINDOW_SAMPLES = 128
SAMPLE_PERIOD_MS = 20.0
ECE_BINS = 10
BOOTSTRAP_REPLICATES = 2000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _closed_audit_source_hashes() -> set[str]:
    registry_path = (
        Path(__file__).resolve().parents[1]
        / "manifests"
        / "pretrained_reuse"
        / "closed-public-test-audits.json"
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    return {
        str(record["source_archive_sha256"]).lower()
        for record in registry["closed_sources"]
        if record.get("official_test_loaded") is True or record.get("reopen_allowed") is False
    }


def _default_audit_receipt_dir() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    root = Path(local_app_data) if local_app_data else Path.home() / ".cache"
    return root / "tiny-omni-decision" / "closed-public-test-audits"


def _claim_audit_source(
    archive_sha256: str,
    *,
    output: Path,
    freeze_path: Path,
    receipt_dir: Path | None = None,
) -> Path:
    archive_sha256 = archive_sha256.lower()
    if archive_sha256 in _closed_audit_source_hashes():
        raise FileExistsError("official test for this source archive is permanently closed")
    registry_dir = receipt_dir or _default_audit_receipt_dir()
    registry_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = registry_dir / f"{archive_sha256}.json"
    receipt = {
        "archive_sha256": archive_sha256,
        "claimed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_directory": str(output.resolve()),
        "candidate_freeze_sha256": _sha256(freeze_path),
        "state": "reserved_before_test_read; never retry this source hash",
    }
    try:
        with receipt_path.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
    except FileExistsError as exc:
        raise FileExistsError(
            "official test audit was already claimed for this source archive"
        ) from exc
    return receipt_path


def _mark_audit_source_loaded(receipt_path: Path) -> None:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["state"] = "test_split_loaded; never retry this source hash"
    receipt["test_loaded_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def _member(archive: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one archive member ending in {suffix!r}")
    return matches[0]


@contextmanager
def _open_uci_har_data(archive_path: Path) -> Iterator[zipfile.ZipFile]:
    """Open the official nested data archive without extracting test members."""
    with zipfile.ZipFile(archive_path) as outer:
        nested = [name for name in outer.namelist() if name.endswith("UCI HAR Dataset.zip")]
        if len(nested) > 1:
            raise ValueError("expected at most one nested UCI HAR data archive")
        if nested:
            payload = io.BytesIO(outer.read(nested[0]))
            with zipfile.ZipFile(payload) as data:
                yield data
        else:
            # Synthetic unit-test fixtures may contain split members directly.
            yield outer


def load_uci_har_split(
    archive: zipfile.ZipFile, split: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read six fixed raw-signal channels plus labels/subject IDs for one split."""
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    x_channels = []
    count = None
    for channel in CHANNELS:
        suffix = f"{split}/Inertial Signals/{channel}_{split}.txt"
        raw = archive.read(_member(archive, suffix))
        values = np.loadtxt(io.BytesIO(raw), dtype=np.float32, ndmin=2)
        if values.shape[1] != WINDOW_SAMPLES:
            raise ValueError(f"unexpected UCI HAR window width for {channel}")
        if count is not None and len(values) != count:
            raise ValueError("UCI HAR channel rows are not aligned")
        count = len(values)
        x_channels.append(values)
    values = np.stack(x_channels, axis=-1)
    y = np.loadtxt(
        io.BytesIO(archive.read(_member(archive, f"{split}/y_{split}.txt"))),
        dtype=np.int64,
        ndmin=1,
    )
    subjects = np.loadtxt(
        io.BytesIO(archive.read(_member(archive, f"{split}/subject_{split}.txt"))),
        dtype=np.int64,
        ndmin=1,
    )
    if (
        values.shape[0] != len(y)
        or len(y) != len(subjects)
        or values.shape[0] == 0
        or not np.isin(y, np.arange(1, 7)).all()
        or not np.isfinite(values).all()
    ):
        raise ValueError("UCI HAR signal, label, subject, or finite-value check failed")
    return values, y, subjects


def split_train_subjects(
    subject_ids: list[int] | tuple[int, ...] | np.ndarray,
    *,
    seed: int = SEED,
    validation_subject_count: int = VALIDATION_SUBJECTS,
) -> tuple[list[int], list[int]]:
    """Deterministically hold out complete subjects from official train only."""
    subjects = [int(value) for value in subject_ids]
    if len(set(subjects)) != len(subjects):
        raise ValueError("subject IDs must be unique")
    if len(subjects) < 2 or not 1 <= validation_subject_count < len(subjects):
        raise ValueError("subject groups must permit nonempty train and validation splits")
    shuffled = sorted(subjects)
    random.Random(seed).shuffle(shuffled)
    validation = sorted(shuffled[:validation_subject_count])
    train = sorted(shuffled[validation_subject_count:])
    if set(train) & set(validation) or set(train) | set(validation) != set(subjects):
        raise RuntimeError("subject split invariant failed")
    return train, validation


def require_disjoint_subjects(
    fit_subjects: list[int] | tuple[int, ...] | np.ndarray,
    evaluation_subjects: list[int] | tuple[int, ...] | np.ndarray,
) -> int:
    """Fail if any participant is shared between fitting and evaluation."""
    fit = {int(value) for value in fit_subjects}
    evaluation = {int(value) for value in evaluation_subjects}
    overlap = fit & evaluation
    if overlap:
        raise ValueError(f"subject overlap between fit and evaluation: {sorted(overlap)}")
    return len(overlap)


def make_option_inputs(device: torch.device) -> tuple[Tensor, Tensor, Tensor, TypedOptionSchema]:
    schema = TypedOptionSchema(enums=(EnumOptionField("activity", ACTIVITIES),))
    from tiny_omni_decision.observations import NativeOption

    options = [
        NativeOption(option_id=activity, attributes={"activity": activity})
        for activity in ACTIVITIES
    ]
    features = schema.encode(options).to(device=device, dtype=torch.float32)
    return (
        features,
        torch.ones((1, len(ACTIVITIES)), dtype=torch.bool, device=device),
        torch.arange(len(ACTIVITIES), dtype=torch.long, device=device),
        schema,
    )


def _normalization(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = values.astype(np.float64).reshape(-1, values.shape[-1])
    mean = flat.mean(axis=0)
    scale = flat.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def _normalize(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    output = ((values - mean[None, None, :]) / scale[None, None, :]).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("normalized windows are nonfinite")
    return output


def _static_features(values: Tensor) -> Tensor:
    return torch.cat(
        (
            values.mean(dim=1),
            values.std(dim=1, unbiased=False),
            values.amin(dim=1),
            values.amax(dim=1),
        ),
        dim=-1,
    )


def _new_model(kind: str) -> nn.Module:
    if kind == "lstm":
        return SensorTemporalDecisionScorer(
            input_channels=len(CHANNELS),
            option_width=len(ACTIVITIES),
            hidden_size=HIDDEN_SIZE,
            scorer_hidden_width=READOUT_HIDDEN_SIZE,
            time_unit_ms=TIME_UNIT_MS,
        )
    if kind == "static":
        return NativeFeatureDecisionScorer(
            feature_width=4 * len(CHANNELS),
            option_width=len(ACTIVITIES),
            hidden_width=READOUT_HIDDEN_SIZE,
        )
    raise ValueError(f"unsupported model kind: {kind}")


def _logits(
    model: nn.Module,
    kind: str,
    values: Tensor,
    option_features: Tensor,
) -> Tensor:
    batch = values.shape[0]
    candidates = option_features.unsqueeze(0).expand(batch, -1, -1)
    option_valid = torch.ones((batch, len(ACTIVITIES)), device=values.device, dtype=torch.bool)
    if kind == "static":
        return model(_static_features(values), candidates, option_valid)
    times = (
        torch.arange(WINDOW_SAMPLES, device=values.device, dtype=values.dtype)
        .mul(SAMPLE_PERIOD_MS)
        .unsqueeze(0)
        .expand(batch, -1)
    )
    valid = torch.ones((batch, WINDOW_SAMPLES), device=values.device, dtype=torch.bool)
    return model(values, times, valid, candidates, option_valid)


def _metrics(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, object]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if (
        probabilities.ndim != 2
        or probabilities.shape != (len(targets), len(ACTIVITIES))
        or len(targets) == 0
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(probabilities.sum(-1), 1.0, atol=1e-6)
    ):
        raise ValueError("finite normalized multiclass probabilities and targets required")
    predicted = probabilities.argmax(axis=1)
    correct = predicted == targets
    class_metrics = {}
    recalls = []
    one_hot = np.eye(len(ACTIVITIES))[targets]
    for index, name in enumerate(ACTIVITIES):
        mask = targets == index
        support = int(mask.sum())
        recall = float(correct[mask].mean()) if support else None
        if support:
            recalls.append(recall)
        class_metrics[name] = {"support": support, "recall": recall}
    confidence = probabilities.max(axis=1)
    bin_ids = np.minimum((confidence * ECE_BINS).astype(int), ECE_BINS - 1)
    ece = 0.0
    for index in range(ECE_BINS):
        mask = bin_ids == index
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return {
        "n": len(targets),
        "accuracy": float(correct.mean()),
        "macro_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "nll": float(
            -np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1.0)).mean()
        ),
        "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ece_10_equal_width": float(ece),
        "mean_confidence": float(confidence.mean()),
        "per_activity": class_metrics,
    }


@torch.inference_mode()
def _predict(
    model: nn.Module,
    kind: str,
    values: np.ndarray,
    targets: np.ndarray,
    subjects: np.ndarray,
    sample_ids: list[str],
    option_features: Tensor,
    *,
    batch_size: int = BATCH_SIZE,
) -> tuple[np.ndarray, dict[str, object], list[dict[str, object]]]:
    model.eval()
    probabilities = []
    for start in range(0, len(values), batch_size):
        x = torch.from_numpy(values[start : start + batch_size])
        probs = _logits(model, kind, x, option_features).softmax(dim=-1).cpu().numpy()
        probabilities.append(probs)
    probability_array = np.concatenate(probabilities)
    metrics = _metrics(probability_array, targets)
    records = []
    for index, row in enumerate(probability_array):
        records.append(
            {
                "sample_id": sample_ids[index],
                "subject_id": int(subjects[index]),
                "target": ACTIVITIES[int(targets[index])],
                "prediction": ACTIVITIES[int(row.argmax())],
                "probabilities": {
                    name: float(value) for name, value in zip(ACTIVITIES, row, strict=True)
                },
            }
        )
    return probability_array, metrics, records


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _save_weights(path: Path, model: nn.Module) -> str:
    state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    save_file(state, str(path))
    return _sha256(path)


def _load_weights(path: Path, kind: str) -> nn.Module:
    model = _new_model(kind)
    model.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    return model.eval()


def _git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
    ).strip()


def _training_config(
    kind: str,
    archive_sha256: str,
    train_subjects: list[int],
    validation_subjects: list[int],
    epochs: int,
    source_commit: str,
    script_sha256: str,
) -> dict[str, object]:
    return {
        "protocol": "uci-har-temporal-decision-v1",
        "source_commit": source_commit,
        "script_sha256": script_sha256,
        "source": "UCI Human Activity Recognition Using Smartphones, dataset id 240, CC BY 4.0",
        "source_doi": "10.24432/C54S4K",
        "source_archive_sha256": archive_sha256,
        "model": kind,
        "seed": SEED,
        "train_subjects": train_subjects,
        "validation_subjects": validation_subjects,
        "official_test_subjects": "not loaded until separate audit phase",
        "channels": list(CHANNELS),
        "window_samples": WINDOW_SAMPLES,
        "sample_rate_hz": 50,
        "sample_period_ms": SAMPLE_PERIOD_MS,
        "normalization": "per-channel mean/std fit on training-subject windows only",
        "options": list(ACTIVITIES),
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "selector": "earliest minimum validation NLL; no test access during fit",
        "compute_device": "cpu",
        "num_workers": 0,
        "pretrained_encoders": False,
        "test_policy": (
            "official public test is a one-time post-freeze reference, not a blind product audit"
        ),
    }


def _fit_model(
    kind: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_subjects: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    validation_subjects: np.ndarray,
    train_ids: list[str],
    validation_ids: list[str],
    option_features: Tensor,
    output: Path,
    normalization: tuple[np.ndarray, np.ndarray],
    epochs: int,
) -> dict[str, object]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = _new_model(kind).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    curve_path = output / f"{kind}-history.jsonl"
    best_nll = float("inf")
    best_epoch = 0
    best_hash = None
    start_time = time.perf_counter()
    train_tensor = torch.from_numpy(train_x)
    target_tensor = torch.from_numpy(train_y.astype(np.int64) - 1)

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        generator = torch.Generator().manual_seed(SEED + epoch)
        order = torch.randperm(len(train_tensor), generator=generator)
        for start in range(0, len(order), BATCH_SIZE):
            indexes = order[start : start + BATCH_SIZE]
            x = train_tensor[indexes]
            y = target_tensor[indexes]
            optimizer.zero_grad(set_to_none=True)
            logits = _logits(model, kind, x, option_features)
            loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite {kind} training loss at epoch {epoch}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"nonfinite {kind} gradient norm at epoch {epoch}")
            optimizer.step()

        _, train_metrics, _ = _predict(
            model,
            kind,
            train_x,
            train_y - 1,
            train_subjects,
            train_ids,
            option_features,
        )
        _, validation_metrics, validation_records = _predict(
            model,
            kind,
            validation_x,
            validation_y - 1,
            validation_subjects,
            validation_ids,
            option_features,
        )
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "elapsed_epoch_seconds": time.perf_counter() - epoch_start,
            "elapsed_run_seconds": time.perf_counter() - start_time,
        }
        with curve_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

        if validation_metrics["nll"] < best_nll:
            best_nll = float(validation_metrics["nll"])
            best_epoch = epoch
            best_hash = _save_weights(output / f"{kind}-best.safetensors", model)
            _write_jsonl(output / f"{kind}-best-validation-predictions.jsonl", validation_records)
        _save_weights(output / f"{kind}-terminal.safetensors", model)

    if best_epoch == 0 or best_hash is None:
        raise RuntimeError(f"no best checkpoint was selected for {kind}")
    selected = _load_weights(output / f"{kind}-best.safetensors", kind)
    _, reloaded_metrics, reloaded_records = _predict(
        selected,
        kind,
        validation_x,
        validation_y - 1,
        validation_subjects,
        validation_ids,
        option_features,
    )
    if reloaded_metrics["nll"] != best_nll:
        raise RuntimeError(f"{kind} best checkpoint reload changed validation NLL")
    _write_jsonl(output / f"{kind}-best-validation-reload.jsonl", reloaded_records)
    mean, scale = normalization
    parameter_count = sum(parameter.numel() for parameter in selected.parameters())
    return {
        "model": kind,
        "best_epoch": best_epoch,
        "best_validation_nll": best_nll,
        "best_validation_metrics": reloaded_metrics,
        "best_checkpoint_sha256": _sha256(output / f"{kind}-best.safetensors"),
        "terminal_checkpoint_sha256": _sha256(output / f"{kind}-terminal.safetensors"),
        "parameter_count": parameter_count,
        "fit_seconds": time.perf_counter() - start_time,
        "normalization_mean": mean.tolist(),
        "normalization_scale": scale.tolist(),
    }


def fit_phase(archive_path: Path, output: Path, *, epochs: int = EPOCHS) -> None:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing experiment directory: {output}")
    output.mkdir(parents=True)
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    source_commit = _git_revision()
    script_sha256 = _sha256(Path(__file__))
    archive_hash = _sha256(archive_path)
    with _open_uci_har_data(archive_path) as archive:
        train_values, train_labels, train_subject_ids = load_uci_har_split(archive, "train")
    subjects = sorted(int(value) for value in np.unique(train_subject_ids))
    train_subjects, validation_subjects = split_train_subjects(subjects)
    train_mask = np.isin(train_subject_ids, train_subjects)
    validation_mask = np.isin(train_subject_ids, validation_subjects)
    raw_train_x, train_y = train_values[train_mask], train_labels[train_mask]
    raw_validation_x, validation_y = train_values[validation_mask], train_labels[validation_mask]
    actual_train_subjects = train_subject_ids[train_mask]
    actual_validation_subjects = train_subject_ids[validation_mask]
    if set(actual_train_subjects.tolist()) != set(train_subjects):
        raise ValueError("training split subject accounting failed")
    if set(actual_validation_subjects.tolist()) != set(validation_subjects):
        raise ValueError("validation split subject accounting failed")
    mean, scale = _normalization(raw_train_x)
    train_x = _normalize(raw_train_x, mean, scale)
    validation_x = _normalize(raw_validation_x, mean, scale)
    train_ids = [f"train-window-{i:05d}" for i in np.flatnonzero(train_mask)]
    validation_ids = [f"train-window-{i:05d}" for i in np.flatnonzero(validation_mask)]

    split_payload = {
        "seed": SEED,
        "split_rule": (
            "subject IDs in official train only; sorted IDs shuffled by "
            "Python Random(seed=17); first four held out for validation"
        ),
        "train_subjects": train_subjects,
        "validation_subjects": validation_subjects,
        "train_sample_ids": train_ids,
        "validation_sample_ids": validation_ids,
        "train_validation_subject_overlap": 0,
        "official_test_split_loaded": False,
    }
    split_bytes = json.dumps(split_payload, sort_keys=True, separators=(",", ":")).encode()
    split_payload["split_manifest_sha256"] = hashlib.sha256(split_bytes).hexdigest()
    _write_json(output / "split-manifest.json", split_payload)
    option_features, _, _, schema = make_option_inputs(torch.device("cpu"))
    _write_json(
        output / "task-schema.json",
        {
            "modality": "sensor",
            "dataset": "UCI HAR",
            "options": list(ACTIVITIES),
            "schema_enum_values": list(schema.enums[0].values),
            "input_channels": list(CHANNELS),
            "target": "activity label for the complete 128-sample 2.56s window",
            "timestamps_ms": "uniform 50Hz sequence starting at 0",
        },
    )
    configs = {}
    results = {}
    for kind in ("static", "lstm"):
        config = _training_config(
            kind,
            archive_hash,
            train_subjects,
            validation_subjects,
            epochs,
            source_commit,
            script_sha256,
        )
        _write_json(output / f"{kind}-effective-config.json", config)
        config_path = output / f"{kind}-effective-config.json"
        config["effective_config_sha256"] = _sha256(config_path)
        configs[kind] = config
        results[kind] = _fit_model(
            kind,
            train_x,
            train_y,
            actual_train_subjects,
            validation_x,
            validation_y,
            actual_validation_subjects,
            train_ids,
            validation_ids,
            option_features,
            output,
            (mean, scale),
            epochs,
        )
    _write_json(
        output / "fit-summary.json",
        {
            "status": "candidate_frozen_before_official_test_read",
            "source_archive_sha256": archive_hash,
            "source_commit": source_commit,
            "script_sha256": script_sha256,
            "started_utc": started_utc,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "split_manifest_sha256": split_payload["split_manifest_sha256"],
            "train_example_count": len(train_y),
            "unique_train_subjects": len(train_subjects),
            "validation_example_count": len(validation_y),
            "unique_validation_subjects": len(validation_subjects),
            "repeat_count_per_model": epochs,
            "unique_train_samples_per_model": len(train_y),
            "static": results["static"],
            "lstm": results["lstm"],
            "effective_configs": configs,
            "official_test_loaded": False,
        },
    )
    freezes = {
        kind: {
            "checkpoint_sha256": results[kind]["best_checkpoint_sha256"],
            "effective_config_sha256": configs[kind]["effective_config_sha256"],
            "best_epoch": results[kind]["best_epoch"],
            "selector": "earliest minimum validation NLL",
        }
        for kind in ("static", "lstm")
    }
    _write_json(
        output / "candidate-freeze.json",
        {
            "frozen_before_test_read": True,
            "source_archive_sha256": archive_hash,
            "source_commit": source_commit,
            "script_sha256": script_sha256,
            "split_manifest_sha256": split_payload["split_manifest_sha256"],
            "candidates": freezes,
        },
    )


def _subject_bootstrap(
    predictions: dict[str, np.ndarray], targets: np.ndarray, subjects: np.ndarray
) -> dict[str, object]:
    unique = sorted(int(value) for value in np.unique(subjects))
    if len(unique) < 2:
        raise ValueError("at least two test subjects are required for clustered uncertainty")
    rng = np.random.default_rng(SEED)
    values = {name: [] for name in ("accuracy", "macro_accuracy")}
    for _ in range(BOOTSTRAP_REPLICATES):
        draw = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(subjects == subject) for subject in draw])
        metric_rows = {}
        for name, prob in predictions.items():
            metric_rows[name] = _metrics(prob[indices], targets[indices])
        for metric in values:
            values[metric].append(metric_rows["lstm"][metric] - metric_rows["static"][metric])
    intervals = {}
    for metric, draws in values.items():
        intervals[metric] = {
            "lstm_minus_static": float(np.mean(draws)),
            "percentile_95_ci": [
                float(np.quantile(draws, 0.025)),
                float(np.quantile(draws, 0.975)),
            ],
        }
    return {
        "method": (
            "paired subject-cluster bootstrap; percentile 95%; resample subjects with replacement"
        ),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": SEED,
        "independent_test_subjects": len(unique),
        "subject_ids": unique,
        "paired_differences": intervals,
    }


def audit_phase(
    archive_path: Path,
    output: Path,
    *,
    receipt_dir: Path | None = None,
) -> None:
    freeze_path = output / "candidate-freeze.json"
    attempt_path = output / "official-test-audit-started.json"
    result_path = output / "official-test-audit.json"
    archive_sha256 = _sha256(archive_path)
    if archive_sha256 in _closed_audit_source_hashes():
        raise FileExistsError("official test for this source archive is permanently closed")
    if not freeze_path.is_file():
        raise FileNotFoundError("frozen candidate record is required before test evaluation")
    if attempt_path.exists() or result_path.exists():
        raise FileExistsError("official test audit is at-most-once; refusing to reopen it")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        not freeze.get("frozen_before_test_read")
        or freeze["source_archive_sha256"] != archive_sha256
    ):
        raise ValueError("candidate freeze or source archive identity mismatch")
    for kind, record in freeze["candidates"].items():
        model_path = output / f"{kind}-best.safetensors"
        config_path = output / f"{kind}-effective-config.json"
        if (
            _sha256(model_path) != record["checkpoint_sha256"]
            or _sha256(config_path) != record["effective_config_sha256"]
        ):
            raise ValueError(f"frozen {kind} candidate files changed")
    audit_receipt_path = _claim_audit_source(
        archive_sha256,
        output=output,
        freeze_path=freeze_path,
        receipt_dir=receipt_dir,
    )
    attempt_path.write_text(
        json.dumps(
            {
                "attempt_started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "archive_sha256": archive_sha256,
                "candidate_freeze_sha256": _sha256(freeze_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with _open_uci_har_data(archive_path) as archive:
        test_values, test_labels, test_subjects = load_uci_har_split(archive, "test")
    _mark_audit_source_loaded(audit_receipt_path)
    fit_summary = json.loads((output / "fit-summary.json").read_text(encoding="utf-8"))
    split_manifest = json.loads((output / "split-manifest.json").read_text(encoding="utf-8"))
    subjects_seen_during_fit = sorted(
        split_manifest["train_subjects"] + split_manifest["validation_subjects"]
    )
    train_test_subject_overlap = require_disjoint_subjects(
        subjects_seen_during_fit, np.unique(test_subjects)
    )
    mean = np.asarray(fit_summary["lstm"]["normalization_mean"], dtype=np.float32)
    scale = np.asarray(fit_summary["lstm"]["normalization_scale"], dtype=np.float32)
    test_x = _normalize(test_values, mean, scale)
    test_ids = [f"official-test-window-{i:05d}" for i in range(len(test_labels))]
    option_features, _, _, _ = make_option_inputs(torch.device("cpu"))
    predictions = {}
    model_reports = {}
    for kind in ("static", "lstm"):
        model = _load_weights(output / f"{kind}-best.safetensors", kind)
        probabilities, metrics, records = _predict(
            model,
            kind,
            test_x,
            test_labels - 1,
            test_subjects,
            test_ids,
            option_features,
        )
        predictions[kind] = probabilities
        model_reports[kind] = {
            "metrics": metrics,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "checkpoint_sha256": _sha256(output / f"{kind}-best.safetensors"),
            "prediction_file": f"{kind}-official-test-predictions.jsonl",
            "prediction_sha256": None,
        }
        pred_path = output / f"{kind}-official-test-predictions.jsonl"
        _write_jsonl(pred_path, records)
        model_reports[kind]["prediction_sha256"] = _sha256(pred_path)
    report = {
        "status": "completed_once_after_candidate_freeze",
        "test_source": "official UCI HAR public test; not a blind global benchmark",
        "source_archive_sha256": _sha256(archive_path),
        "candidate_freeze_sha256": _sha256(freeze_path),
        "official_test_windows": len(test_labels),
        "unique_test_subjects": sorted(int(value) for value in np.unique(test_subjects)),
        "subjects_seen_during_fit": subjects_seen_during_fit,
        "train_test_subject_overlap": train_test_subject_overlap,
        "models": model_reports,
        "paired_subject_bootstrap": _subject_bootstrap(predictions, test_labels - 1, test_subjects),
        "test_loaded_for_first_time_after_freeze": True,
        "no_test_based_model_selection_or_retraining": True,
    }
    _write_json(result_path, report)


def _environment() -> dict[str, object]:
    return {
        "source_commit": _git_revision(),
        "script_sha256": _sha256(Path(__file__)),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "cpu_count": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("fit", "audit"), required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    if args.phase == "fit":
        fit_phase(args.archive, args.output_dir, epochs=args.epochs)
        _write_json(args.output_dir / "environment.json", _environment())
    else:
        audit_phase(args.archive, args.output_dir)


if __name__ == "__main__":
    main()
