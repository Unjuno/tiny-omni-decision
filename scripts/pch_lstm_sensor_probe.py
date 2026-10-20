"""Compare a compact temporal sensor readout with the frozen MOMENT probe."""

from __future__ import annotations

import argparse
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
    SEED,
    _metric_summary,
    build_records,
    participant_bootstrap_comparison,
    preflight_inventory,
    sha256_bytes,
    sha256_file,
    subject_split_for_record,
)

CONFIG = {
    "model": "one-layer LSTM over calibrated 2.5-second triaxial windows",
    "hidden_size": 64,
    "num_layers": 1,
    "dropout": 0.0,
    "optimizer": "AdamW",
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "batch_size": 64,
    "max_epochs": 100,
    "early_stopping_patience": 10,
    "min_delta": 1e-5,
    "gradient_clip_norm": 1.0,
    "seed": SEED,
    "selection": "minimum participant-aggregated validation NLL",
    "record_aggregation": "mean window logits; per-record equal loss weight",
    "normalization": "fit-only global per-axis mean and standard deviation",
}
MOMENT_REFERENCE_RESULT_SHA256 = "e14255295125421708bdcd82edeaa20364b8698b46ac09f279b57da19ed92633"
MOMENT_REFERENCE_PREDICTIONS_SHA256 = (
    "78d5994e254829c8d3b55eb8ae591c38445daa7556d65ad2f923cffe3e81c93f"
)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(torch, path: Path, value: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f"{path.stem}-", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _records(archive_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with zipfile.ZipFile(archive_path) as archive:
        return build_records(archive)


def prepare_preflight(
    archive_path: Path,
    reference_predictions: Path,
    reference_result: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"attempt output already exists: {output_dir}")
    if archive_path.stat().st_size != ARCHIVE_SIZE or sha256_file(archive_path) != ARCHIVE_SHA256:
        raise ValueError("PCH archive size/SHA-256 differs from the acquired official source")
    if sha256_file(reference_result) != MOMENT_REFERENCE_RESULT_SHA256:
        raise ValueError("MOMENT reference result differs from the frozen attempt-01 receipt")
    if sha256_file(reference_predictions) != MOMENT_REFERENCE_PREDICTIONS_SHA256:
        raise ValueError("MOMENT reference predictions differ from the frozen attempt-01 receipt")
    records, split = _records(archive_path)
    inventory = preflight_inventory(records, split, batch_size=CONFIG["batch_size"])
    validation = [
        record for record in records if subject_split_for_record(record, split) == "validation"
    ]
    reference_rows = [
        json.loads(line) for line in reference_predictions.read_text(encoding="utf-8").splitlines()
    ]
    if [row.get("sample_id") for row in reference_rows] != [
        record["sample_id"] for record in validation
    ]:
        raise ValueError("MOMENT reference predictions do not match fixed validation ID order")
    if any(
        row.get("target_id") != record["label"]
        for row, record in zip(reference_rows, validation, strict=True)
    ):
        raise ValueError("MOMENT reference labels differ from fixed validation labels")
    if any(
        len(row.get("moment_probabilities", [])) != len(ACTIVITY_IDS)
        or not np.isfinite(row["moment_probabilities"]).all()
        for row in reference_rows
    ):
        raise ValueError("MOMENT reference probabilities are invalid")
    result = {
        "status": "PASS_PCH_LSTM_PREFLIGHT",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "archive_sha256": ARCHIVE_SHA256,
        "archive_bytes": ARCHIVE_SIZE,
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "moment_reference_predictions_path": str(reference_predictions.resolve()),
        "moment_reference_predictions_sha256": sha256_file(reference_predictions),
        "moment_reference_result_path": str(reference_result.resolve()),
        "moment_reference_result_sha256": sha256_file(reference_result),
        "config": CONFIG,
        "split_metadata": split,
        "inventory": inventory,
        "window_order_sha256": sha256_bytes(
            "\n".join(
                f"{record['sample_id']}:{index}"
                for record in records
                for index in range(record["window_count"])
            ).encode()
        ),
        "fit_unique_records": len(
            [r for r in records if subject_split_for_record(r, split) == "fit"]
        ),
        "validation_unique_records": len(
            [r for r in records if subject_split_for_record(r, split) == "validation"]
        ),
        "fit_validation_windows": sum(r["window_count"] for r in records),
        "heldout_payload_opened": False,
    }
    output_dir.mkdir(parents=True)
    _atomic_json(output_dir / "preflight.json", result)
    return result


def _window_arrays(records: list[dict[str, Any]], split: dict[str, Any]) -> dict[str, Any]:
    fit = [r for r in records if subject_split_for_record(r, split) == "fit"]
    validation = [r for r in records if subject_split_for_record(r, split) == "validation"]
    label_to_id = {label: index for index, label in enumerate(ACTIVITY_IDS)}

    def flatten(
        selected: list[dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        windows = np.concatenate([record["windows"] for record in selected], axis=0)
        targets = np.concatenate(
            [np.full(record["window_count"], label_to_id[record["label"]]) for record in selected]
        ).astype(np.int64)
        owners = np.concatenate(
            [np.full(record["window_count"], index) for index, record in enumerate(selected)]
        ).astype(np.int64)
        counts = np.asarray([record["window_count"] for record in selected], dtype=np.int64)
        return windows, targets, owners, counts

    return {
        "fit_records": fit,
        "validation_records": validation,
        "fit": flatten(fit),
        "validation": flatten(validation),
    }


def _build_model(torch, hidden_size: int, num_layers: int):
    from torch import nn

    class WindowLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(3, hidden_size, num_layers=num_layers, batch_first=True)
            self.classifier = nn.Linear(hidden_size, len(ACTIVITY_IDS))

        def forward(self, windows):
            _, (hidden, _) = self.lstm(windows)
            return self.classifier(hidden[-1])

    return WindowLSTM()


def _record_logits(torch, model, windows, owners, record_count, mean, scale, device, batch_size):
    model.eval()
    logits_by_record: list[list[np.ndarray]] = [[] for _ in range(record_count)]
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            end = min(start + batch_size, len(windows))
            batch = torch.as_tensor(windows[start:end], dtype=torch.float32, device=device)
            batch = (batch - mean) / scale
            output = model(batch.transpose(1, 2).contiguous()).float().cpu().numpy()
            for owner, logits in zip(owners[start:end], output, strict=True):
                logits_by_record[int(owner)].append(logits)
    return np.stack([np.mean(rows, axis=0) for rows in logits_by_record]).astype(np.float32)


def _evaluate(torch, logits: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    probabilities = torch.as_tensor(logits).softmax(dim=-1).numpy()
    metrics = _metric_summary(probabilities, targets)
    return probabilities, metrics


def run_experiment(
    *,
    archive_path: Path,
    reference_predictions: Path,
    reference_result: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if not output_dir.is_dir() or {path.name for path in output_dir.iterdir()} != {
        "preflight.json"
    }:
        raise ValueError("attempt output must contain only the immutable preflight.json")
    preflight_path = output_dir / "preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("script_sha256") != sha256_file(Path(__file__).resolve()):
        raise ValueError("preflight does not match this exact experiment script")
    if archive_path.stat().st_size != ARCHIVE_SIZE or sha256_file(archive_path) != ARCHIVE_SHA256:
        raise ValueError("PCH archive size/SHA-256 differs from the acquired official source")
    if (
        str(reference_predictions.resolve()) != preflight["moment_reference_predictions_path"]
        or sha256_file(reference_predictions) != preflight["moment_reference_predictions_sha256"]
        or str(reference_result.resolve()) != preflight["moment_reference_result_path"]
        or sha256_file(reference_result) != MOMENT_REFERENCE_RESULT_SHA256
    ):
        raise ValueError("MOMENT reference artifacts differ from the immutable preflight")

    from scripts.pch_moment_sensor_probe import gpu_preflight

    hotgpu = gpu_preflight(dict(os.environ))
    import torch
    from torch.nn.utils import clip_grad_norm_

    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    config_hash = sha256_bytes(json.dumps(CONFIG, sort_keys=True, separators=(",", ":")).encode())
    (output_dir / "effective-config.json").write_text(
        json.dumps(CONFIG, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _atomic_json(
        output_dir / "run-metadata.json",
        {
            "status": "RUNNING",
            "started_at_utc": started_at,
            "config_sha256": config_hash,
            "preflight_sha256": sha256_file(preflight_path),
            "archive_sha256": ARCHIVE_SHA256,
            "reference_predictions_sha256": preflight["moment_reference_predictions_sha256"],
        },
    )
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")

    records, split = _records(archive_path)
    inventory = preflight_inventory(records, split, batch_size=CONFIG["batch_size"])
    if split != preflight["split_metadata"] or inventory != preflight["inventory"]:
        raise ValueError("runtime split/sample inventory differs from immutable preflight")
    data = _window_arrays(records, split)
    fit_records = data["fit_records"]
    validation_records = data["validation_records"]
    fit_windows, fit_targets, fit_owners, fit_counts = data["fit"]
    val_windows, val_targets, val_owners, _ = data["validation"]
    fit_labels = np.asarray([ACTIVITY_IDS.index(r["label"]) for r in fit_records], dtype=np.int64)
    val_labels = np.asarray(
        [ACTIVITY_IDS.index(r["label"]) for r in validation_records], dtype=np.int64
    )
    val_subjects = [r["subject_id"] for r in validation_records]
    val_ids = [r["sample_id"] for r in validation_records]
    id_hash = sha256_bytes("\n".join(val_ids).encode())
    if id_hash != inventory["validation_sample_id_order_sha256"]:
        raise ValueError("validation sample order changed")

    reference = [
        json.loads(line) for line in reference_predictions.read_text(encoding="utf-8").splitlines()
    ]
    if len(reference) != len(validation_records):
        raise ValueError("MOMENT reference prediction count differs from validation records")
    reference_by_id = {row["sample_id"]: row for row in reference}
    if set(reference_by_id) != set(val_ids):
        raise ValueError("MOMENT reference validation IDs differ from this split")
    if any(
        reference_by_id[sample_id]["target_id"] != record["label"]
        for sample_id, record in zip(val_ids, validation_records, strict=True)
    ):
        raise ValueError("MOMENT reference labels differ from fixed validation labels")
    moment_probs = np.asarray(
        [reference_by_id[sample_id]["moment_probabilities"] for sample_id in val_ids],
        dtype=np.float64,
    )
    if (
        moment_probs.shape != (len(val_ids), len(ACTIVITY_IDS))
        or not np.isfinite(moment_probs).all()
    ):
        raise ValueError("MOMENT reference probabilities have invalid shape or values")

    axis_mean = fit_windows.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    axis_scale = fit_windows.std(axis=(0, 2), keepdims=True).astype(np.float32)
    axis_scale = np.maximum(axis_scale, 1e-6)
    mean = torch.as_tensor(axis_mean, device=device)
    scale = torch.as_tensor(axis_scale, device=device)

    model = _build_model(torch, CONFIG["hidden_size"], CONFIG["num_layers"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"]
    )
    count_per_record = torch.as_tensor(fit_counts, dtype=torch.float32, device=device)
    best_nll = float("inf")
    best_epoch = 0
    best_state = None
    best_metrics = None
    best_probabilities = None
    history = []
    patience = int(CONFIG["early_stopping_patience"])
    batch_size = int(CONFIG["batch_size"])
    torch.cuda.reset_peak_memory_stats()
    training_started = time.perf_counter()

    for epoch in range(1, int(CONFIG["max_epochs"]) + 1):
        model.train()
        order = torch.randperm(len(fit_windows), device="cpu").numpy()
        epoch_loss = 0.0
        epoch_weight = 0.0
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            batch = torch.as_tensor(fit_windows[selected], dtype=torch.float32, device=device)
            batch = (batch - mean) / scale
            batch = batch.transpose(1, 2).contiguous()
            targets = torch.as_tensor(fit_targets[selected], dtype=torch.long, device=device)
            owners = torch.as_tensor(fit_owners[selected], dtype=torch.long, device=device)
            weights = count_per_record[owners].reciprocal()
            weights = weights / weights.mean().clamp_min(1e-12)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            losses = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
            loss = (losses * weights).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite LSTM training loss at epoch {epoch}")
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), float(CONFIG["gradient_clip_norm"]))
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"nonfinite LSTM gradient norm at epoch {epoch}")
            optimizer.step()
            epoch_loss += float(loss.detach().cpu()) * len(selected)
            epoch_weight += len(selected)

        train_logits = _record_logits(
            torch,
            model,
            fit_windows,
            fit_owners,
            len(fit_records),
            mean,
            scale,
            device,
            batch_size,
        )
        validation_logits = _record_logits(
            torch,
            model,
            val_windows,
            val_owners,
            len(validation_records),
            mean,
            scale,
            device,
            batch_size,
        )
        train_probabilities, train_metrics = _evaluate(torch, train_logits, fit_labels)
        validation_probabilities, validation_metrics = _evaluate(
            torch, validation_logits, val_labels
        )
        validation_nll = float(validation_metrics["nll"])
        history.append(
            {
                "epoch": epoch,
                "window_weighted_train_ce": epoch_loss / max(epoch_weight, 1),
                "train_record_accuracy": train_metrics["accuracy"],
                "validation_record_nll": validation_nll,
                "validation_record_accuracy": validation_metrics["accuracy"],
                "validation_record_brier": validation_metrics["brier"],
                "validation_record_ece_15": validation_metrics["ece_15"],
                "last_batch_grad_norm": float(grad_norm.detach().cpu()),
            }
        )
        with (output_dir / "training-history.jsonl").open("a", encoding="utf-8") as history_file:
            history_file.write(json.dumps(history[-1], sort_keys=True) + "\n")
        if validation_nll < best_nll - float(CONFIG["min_delta"]):
            best_nll = validation_nll
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            best_metrics = validation_metrics
            best_probabilities = validation_probabilities.copy()
            _atomic_torch_save(
                torch,
                output_dir / "lstm_best.pt",
                {
                    "state_dict": best_state,
                    "axis_mean": torch.as_tensor(axis_mean),
                    "axis_scale": torch.as_tensor(axis_scale),
                    "selected_epoch": best_epoch,
                    "class_ids": list(ACTIVITY_IDS),
                },
            )
        elif epoch - best_epoch >= patience:
            break

    training_seconds = time.perf_counter() - training_started
    finished_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    if best_state is None or best_metrics is None or best_probabilities is None:
        raise RuntimeError("no finite best validation checkpoint was selected")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if not (output_dir / "lstm_best.pt").is_file():
        raise RuntimeError("selected best checkpoint was not durably written")
    _atomic_torch_save(
        torch,
        output_dir / "lstm_final.pt",
        {
            "state_dict": finished_state,
            "axis_mean": torch.as_tensor(axis_mean),
            "axis_scale": torch.as_tensor(axis_scale),
            "final_epoch": history[-1]["epoch"],
            "class_ids": list(ACTIVITY_IDS),
        },
    )
    (output_dir / "training-history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "effective-config.json").write_text(
        json.dumps(CONFIG, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    checkpoint = torch.load(output_dir / "lstm_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    mean = checkpoint["axis_mean"].to(device)
    scale = checkpoint["axis_scale"].to(device)
    reloaded_logits = _record_logits(
        torch,
        model,
        val_windows,
        val_owners,
        len(validation_records),
        mean,
        scale,
        device,
        batch_size,
    )
    reloaded_probs, reload_metrics = _evaluate(torch, reloaded_logits, val_labels)
    if not np.array_equal(best_probabilities, reloaded_probs):
        raise RuntimeError("best checkpoint reload did not reproduce final selected probabilities")
    if reload_metrics != best_metrics:
        raise RuntimeError("best checkpoint reload metrics differ from the selected metrics")

    moment_metrics = _metric_summary(moment_probs, val_labels)
    comparison = participant_bootstrap_comparison(
        reloaded_probs,
        moment_probs,
        val_labels,
        val_subjects,
        replicates=5000,
    )
    predictions = []
    for index, record in enumerate(validation_records):
        predictions.append(
            {
                "sample_id": record["sample_id"],
                "subject_id": record["subject_id"],
                "target_id": record["label"],
                "window_count": record["window_count"],
                "lstm_logits": reloaded_logits[index].tolist(),
                "lstm_probabilities": reloaded_probs[index].tolist(),
                "moment_probabilities": moment_probs[index].tolist(),
            }
        )
    (output_dir / "validation-predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions), encoding="utf-8"
    )
    torch.cuda.synchronize()
    result = {
        "status": "PASS_PCH_LSTM_SENSOR_DEV_COMPARISON",
        "task": "direct text-free 18-way wearable activity decision per recording",
        "scope": "development validation only; single subject split; not product quality",
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "source_commit": preflight["source_commit"],
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "preflight_sha256": sha256_file(preflight_path),
        "archive_sha256": ARCHIVE_SHA256,
        "moment_reference_predictions_sha256": preflight["moment_reference_predictions_sha256"],
        "moment_reference_result_sha256": preflight["moment_reference_result_sha256"],
        "validation_id_order_sha256": id_hash,
        "training": {
            "records": len(fit_records),
            "unique_participants": len({r["subject_id"] for r in fit_records}),
            "windows": len(fit_windows),
            "optimizer_updates": sum(
                (len(fit_windows) + batch_size - 1) // batch_size for _ in history
            ),
            "selected_epoch": best_epoch,
            "stopping_epoch": history[-1]["epoch"],
            "elapsed_seconds": training_seconds,
            "history": history,
            "early_stopped": history[-1]["epoch"] < int(CONFIG["max_epochs"]),
        },
        "validation": {
            "records": len(validation_records),
            "unique_participants": len(set(val_subjects)),
            "windows": len(val_windows),
            "metrics": reload_metrics,
            "moment_frozen_reference_metrics": moment_metrics,
            "participant_cluster_bootstrap_lstm_minus_moment": comparison,
            "best_checkpoint_reload_exact_probabilities": True,
        },
        "model": {
            "architecture": CONFIG["model"],
            "trainable_parameters": parameter_count,
            "total_parameters": parameter_count,
            "input": "fit-normalized calibrated 3-axis acceleration; 128 samples/window at 51.2Hz",
            "aggregation": CONFIG["record_aggregation"],
        },
        "compute": {
            "device": torch.cuda.get_device_name(device),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "hotgpu_preflight": hotgpu,
            "elapsed_seconds_total": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "cloud_cost_usd": 0,
        },
        "split": split,
        "heldout_payload_opened": False,
        "sealed_audit_loaded": False,
        "checkpoint_selection": (
            "development validation minimum recording-level NLL; no heldout data"
        ),
        "config_sha256": config_hash,
    }
    _atomic_json(output_dir / "result.json", result)
    _atomic_json(
        output_dir / "run-metadata.json",
        {
            "status": "PASS_COMPLETE",
            "started_at_utc": started_at,
            "finished_at_utc": result["finished_at_utc"],
            "config_sha256": config_hash,
            "preflight_sha256": result["preflight_sha256"],
            "result_sha256": sha256_file(output_dir / "result.json"),
            "best_epoch": best_epoch,
            "stopping_epoch": history[-1]["epoch"],
        },
    )
    (output_dir / "command.txt").write_text(
        "python -m scripts.pch_lstm_sensor_probe run --archive <PCH_Data_2017.zip> "
        "--reference-predictions <attempt-01/validation-predictions.jsonl> "
        "--reference-result <attempt-01/result.json> "
        f"--output-dir {output_dir}\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--archive", type=Path, required=True)
    prepare.add_argument("--reference-predictions", type=Path, required=True)
    prepare.add_argument("--reference-result", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--archive", type=Path, required=True)
    run.add_argument("--reference-predictions", type=Path, required=True)
    run.add_argument("--reference-result", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(
            json.dumps(
                prepare_preflight(
                    args.archive,
                    args.reference_predictions,
                    args.reference_result,
                    args.output_dir,
                ),
                indent=2,
            )
        )
    else:
        print(
            json.dumps(
                run_experiment(
                    archive_path=args.archive,
                    reference_predictions=args.reference_predictions,
                    reference_result=args.reference_result,
                    output_dir=args.output_dir,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
