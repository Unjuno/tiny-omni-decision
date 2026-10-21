"""One-time inference-only comparison on the frozen PCH participant holdout."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from scripts.pch_moment_sensor_probe import (
    ACTIVITY_IDS,
    ARCHIVE_SHA256,
    ARCHIVE_SIZE,
    MOMENT_MODEL_REVISION,
    MOMENT_SOURCE_COMMIT,
    MOMENT_WEIGHTS_SHA256,
    assign_subject_splits,
    canonical_activity,
    decode_recording,
    resample_and_window,
    sha256_bytes,
    sha256_file,
    stratify_validation_with_fit_only_labels,
    subject_id_from_member,
)

ARCHIVE = Path(r"C:\CodexArtifacts\pretrained-reuse-r2\pch-data-2017-attempt-01\PCH_Data_2017.zip")
MOMENT_DIR = Path(r"C:\CodexArtifacts\pretrained-reuse-r1\moment-small-sensor-r1-attempt-01\model")
MOMENT_SOURCE = Path(
    r"C:\CodexArtifacts\pretrained-reuse-r1\moment-small-sensor-r1-attempt-01\moment-source"
)
MOMENT_HEAD = Path(
    r"C:\CodexArtifacts\pretrained-reuse-r2\pch-moment-activity-probe-attempt-01\moment_frozen-best.pt"
)
MOMENT_HEAD_SHA256 = "1abec7fd55857f569b8c4ce3990477aabd02c694adb529682d6f66ff3bd1a895"
MOMENT_RESULT = Path(
    r"C:\CodexArtifacts\pretrained-reuse-r2\pch-moment-activity-probe-attempt-01\result.json"
)
MOMENT_RESULT_SHA256 = "e14255295125421708bdcd82edeaa20364b8698b46ac09f279b57da19ed92633"
LSTM_CHECKPOINT = Path(
    r"C:\CodexArtifacts\pretrained-reuse-r2\pch-lstm-sensor-attempt-04\lstm_best.pt"
)
LSTM_CHECKPOINT_SHA256 = "b98cd679280db514a091076bbc7a688ce1e1512c912e9553a114a7ead168facd"
LSTM_RESULT = Path(r"C:\CodexArtifacts\pretrained-reuse-r2\pch-lstm-sensor-attempt-04\result.json")
LSTM_RESULT_SHA256 = "829469373588807fea22ada0afb019fc5895b264ca5845f5227f3e57320b8606"
LSTM_PREFLIGHT = Path(
    r"C:\CodexArtifacts\pretrained-reuse-r2\pch-lstm-sensor-attempt-04\preflight.json"
)
HELDOUT_SUBJECT_SHA256 = "4dc5a3e76d3fcb19ef2ff19a18c92db88269a077ca6b180866e99c4be2bd901"
BOOTSTRAP_DRAWS = 10_000
SEED = 17
ECE_BINS = 15


def _names_hash(values: list[str]) -> str:
    return sha256_bytes("\n".join(sorted(values)).encode())


def verify_holdout_members(
    members_by_split: dict[str, list[dict[str, Any]]],
    *,
    expected_subject_ids: list[str],
    expected_subject_hash: str | None,
) -> dict[str, Any]:
    subjects = {
        split: {str(row["subject_id"]) for row in members_by_split[split]}
        for split in ("fit", "validation", "heldout")
    }
    for left, right in (("fit", "validation"), ("fit", "heldout"), ("validation", "heldout")):
        if subjects[left] & subjects[right]:
            raise ValueError(f"{left} and {right} participant overlap")
    actual = sorted(subjects["heldout"])
    if actual != sorted(expected_subject_ids):
        raise ValueError("heldout participant set differs from the frozen set")
    digest = _names_hash(actual)
    if expected_subject_hash is not None and digest != expected_subject_hash:
        raise ValueError("heldout participant hash differs from the frozen hash")
    return {
        "record_count": len(members_by_split["heldout"]),
        "subject_ids": actual,
        "subject_ids_sha256": digest,
    }


def aggregate_window_logits(
    logits: np.ndarray, record_indices: np.ndarray, *, record_count: int
) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    owners = np.asarray(record_indices, dtype=np.int64)
    if values.ndim != 2 or owners.shape != (len(values),):
        raise ValueError("window logits and recording indices must align")
    if record_count < 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite nonempty logits and positive recording count are required")
    if owners.min() < 0 or owners.max() >= record_count:
        raise ValueError("record index outside the fixed recording order")
    groups = [values[owners == index] for index in range(record_count)]
    if any(not len(group) for group in groups):
        raise ValueError("every recording needs at least one complete window")
    return np.stack([group.mean(axis=0) for group in groups])


def verify_content_separation(
    heldout_content: list[str],
    heldout_signals: list[str],
    prior_content: list[str],
    prior_signals: list[str],
) -> None:
    if len(set(heldout_content)) != len(heldout_content) or set(heldout_content) & set(
        prior_content
    ):
        raise ValueError("raw content overlap or duplicate recording detected across splits")
    if len(set(heldout_signals)) != len(heldout_signals) or set(heldout_signals) & set(
        prior_signals
    ):
        raise ValueError("normalized signal overlap or duplicate recording detected across splits")


def _probabilities(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("logits must be a finite matrix")
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def summarize_logits(
    logits: np.ndarray, targets: np.ndarray, *, class_count: int
) -> dict[str, Any]:
    scores = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    if (
        scores.shape != (len(labels), class_count)
        or not len(labels)
        or not np.isfinite(scores).all()
    ):
        raise ValueError("finite logits, targets and fixed class count must agree")
    if np.any(labels < 0) or np.any(labels >= class_count):
        raise ValueError("target outside the fixed class taxonomy")
    probs = _probabilities(scores)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = pred == labels
    per_class = []
    for label in range(class_count):
        mask = labels == label
        per_class.append(
            {
                "class_id": label,
                "class_name": ACTIVITY_IDS[label],
                "support": int(mask.sum()),
                "accuracy": float((pred[mask] == label).mean()) if mask.any() else None,
            }
        )
    recalls = [row["accuracy"] for row in per_class if row["accuracy"] is not None]
    one_hot = np.eye(class_count)[labels]
    nll = -np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-15, 1.0))
    ece = 0.0
    for index in range(ECE_BINS):
        low, high = index / ECE_BINS, (index + 1) / ECE_BINS
        mask = (conf >= low) & (conf < high if index < ECE_BINS - 1 else conf <= high)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return {
        "count": len(labels),
        "accuracy": float(correct.mean()),
        "macro_accuracy": float(np.mean(recalls)),
        "nll": float(nll.mean()),
        "brier": float(np.square(probs - one_hot).sum(axis=1).mean()),
        "ece_15": float(ece),
        "mean_confidence": float(conf.mean()),
        "class_support": np.bincount(labels, minlength=class_count).tolist(),
        "per_class": per_class,
    }


def paired_subject_bootstrap(
    logits_a, logits_b, targets, subject_ids, *, draws=BOOTSTRAP_DRAWS, seed=SEED
):
    a, b = _probabilities(logits_a), _probabilities(logits_b)
    labels = np.asarray(targets, dtype=np.int64)
    subjects = np.asarray(subject_ids).astype(str)
    if len(a) != len(b) or len(a) != len(labels) or len(a) != len(subjects):
        raise ValueError("paired predictions, targets and subject IDs must align")
    groups = sorted(set(subjects.tolist()))
    if len(groups) < 2 or draws < 100:
        raise ValueError("at least two groups and 100 bootstrap draws are required")
    indices = {group: np.flatnonzero(subjects == group) for group in groups}
    rng = np.random.default_rng(seed)
    deltas = {"accuracy": [], "nll": [], "brier": []}
    for _ in range(draws):
        selected = rng.choice(groups, len(groups), replace=True)
        rows = np.concatenate([indices[group] for group in selected])
        correct_a, correct_b = (
            a[rows].argmax(axis=1) == labels[rows],
            b[rows].argmax(axis=1) == labels[rows],
        )
        nll_a = -np.log(np.clip(a[rows, labels[rows]], 1e-15, 1.0))
        nll_b = -np.log(np.clip(b[rows, labels[rows]], 1e-15, 1.0))
        one_hot = np.eye(a.shape[1])[labels[rows]]
        brier_a = np.square(a[rows] - one_hot).sum(axis=1)
        brier_b = np.square(b[rows] - one_hot).sum(axis=1)
        deltas["accuracy"].append(float((correct_a.astype(float) - correct_b.astype(float)).mean()))
        deltas["nll"].append(float((nll_a - nll_b).mean()))
        deltas["brier"].append(float((brier_a - brier_b).mean()))

    def interval(values: list[float]) -> list[float]:
        return [float(x) for x in np.quantile(values, [0.025, 0.975])]

    return {
        "method": "paired participant-cluster percentile bootstrap",
        "cluster_count": len(groups),
        "draws": draws,
        "seed": seed,
        "direction": "model_a_minus_model_b",
        "model_a": "window LSTM",
        "model_b": "frozen MOMENT plus selected linear head",
        "accuracy_delta": float(np.mean(deltas["accuracy"])),
        "accuracy_delta_95_percent_interval": interval(deltas["accuracy"]),
        "nll_delta": float(np.mean(deltas["nll"])),
        "nll_delta_95_percent_interval": interval(deltas["nll"]),
        "brier_delta": float(np.mean(deltas["brier"])),
        "brier_delta_95_percent_interval": interval(deltas["brier"]),
    }


def _pins() -> dict[str, str]:
    pins = {
        "archive": sha256_file(ARCHIVE),
        "moment_weights": sha256_file(MOMENT_DIR / "model.safetensors"),
        "moment_head": sha256_file(MOMENT_HEAD),
        "moment_result": sha256_file(MOMENT_RESULT),
        "lstm_checkpoint": sha256_file(LSTM_CHECKPOINT),
        "lstm_result": sha256_file(LSTM_RESULT),
        "lstm_preflight": sha256_file(LSTM_PREFLIGHT),
    }
    expected = {
        "archive": ARCHIVE_SHA256,
        "moment_weights": MOMENT_WEIGHTS_SHA256,
        "moment_head": MOMENT_HEAD_SHA256,
        "moment_result": MOMENT_RESULT_SHA256,
        "lstm_checkpoint": LSTM_CHECKPOINT_SHA256,
        "lstm_result": LSTM_RESULT_SHA256,
    }
    if any(pins[key] != value for key, value in expected.items()):
        raise ValueError("frozen archive/model/checkpoint hash mismatch")
    return pins


def prepare_preflight(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"fresh output directory required: {output_dir}")
    if ARCHIVE.stat().st_size != ARCHIVE_SIZE:
        raise ValueError("PCH archive size differs from the frozen source")
    with zipfile.ZipFile(ARCHIVE) as archive:
        names = sorted(
            info.filename
            for info in archive.infolist()
            if not info.is_dir() and info.filename.casefold().endswith(".csv")
        )
        initial, _ = assign_subject_splits(names)
        split = stratify_validation_with_fit_only_labels(names, initial)
        rows = {
            key: [
                {"subject_id": subject_id_from_member(name), "member_name": name}
                for name in names
                if split[subject_id_from_member(name)] == key
            ]
            for key in ("fit", "validation", "heldout")
        }
        content_hashes = {"fit": [], "validation": []}
        signal_hashes = {"fit": [], "validation": []}
        for key in ("fit", "validation"):
            for row in rows[key]:
                raw = archive.read(row["member_name"])
                timestamps, axes, _ = decode_recording(raw)
                windows = resample_and_window(timestamps, axes)
                content_hashes[key].append(sha256_bytes(raw))
                signal_hashes[key].append(sha256_bytes(windows.tobytes()))
    if set(content_hashes["fit"]) & set(content_hashes["validation"]):
        raise ValueError("raw recording content overlaps between fit and validation")
    if set(signal_hashes["fit"]) & set(signal_hashes["validation"]):
        raise ValueError("normalized sensor signals overlap between fit and validation")
    expected_subjects = sorted({row["subject_id"] for row in rows["heldout"]})
    check = verify_holdout_members(
        rows, expected_subject_ids=expected_subjects, expected_subject_hash=HELDOUT_SUBJECT_SHA256
    )
    if len(names) != 423 or check["record_count"] != 84 or len(expected_subjects) != 28:
        raise ValueError("PCH holdout member inventory changed")
    report = {
        "status": "PASS_PCH_HOLDOUT_EVAL_PREREGISTERED",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "archive_sha256": ARCHIVE_SHA256,
        "archive_bytes": ARCHIVE_SIZE,
        "member_count": len(names),
        "split_method": (
            "seed-17 participant hash; fit-only deterministic validation coverage swaps; "
            "heldout assignment unchanged"
        ),
        "record_counts": {key: len(value) for key, value in rows.items()},
        "subject_counts": {
            key: len({row["subject_id"] for row in value}) for key, value in rows.items()
        },
        "subject_ids_sha256": {
            key: _names_hash(sorted({row["subject_id"] for row in value}))
            for key, value in rows.items()
        },
        "heldout_member_names_sha256": _names_hash([row["member_name"] for row in rows["heldout"]]),
        "nonheldout_content_hashes": content_hashes,
        "nonheldout_normalized_signal_hashes": signal_hashes,
        "model_pins": _pins(),
        "moment_model_revision": MOMENT_MODEL_REVISION,
        "moment_source_commit": MOMENT_SOURCE_COMMIT,
        "candidate_selection": {
            "lstm_selected_epoch": 31,
            "moment_selected_epoch": 12,
            "selected_using": "existing development validation only",
        },
        "candidates": [
            "selected seed-17 window LSTM checkpoint from attempt-04",
            "frozen MOMENT encoder and selected linear head from attempt-01",
        ],
        "policy": (
            "inference only; no training, calibration, thresholding, checkpoint or model "
            "selection; execute once"
        ),
        "metrics": [
            "Accuracy",
            "present-class macro Accuracy",
            "NLL",
            "multiclass Brier",
            "ECE-15",
            "mean confidence",
            "per-class support/accuracy",
        ],
        "bootstrap": {
            "method": "paired participant-cluster percentile",
            "draws": BOOTSTRAP_DRAWS,
            "seed": SEED,
        },
        "heldout_labels_or_payload_opened": False,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "preflight.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _heldout_records():
    records = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        names = sorted(
            info.filename
            for info in archive.infolist()
            if not info.is_dir() and info.filename.casefold().endswith(".csv")
        )
        initial, _ = assign_subject_splits(names)
        split = stratify_validation_with_fit_only_labels(names, initial)
        for name in names:
            subject = subject_id_from_member(name)
            if split[subject] != "heldout":
                continue
            label = canonical_activity(name)
            raw = archive.read(name)
            timestamps, axes, _ = decode_recording(raw)
            windows = resample_and_window(timestamps, axes)
            records.append(
                {
                    "sample_id": sha256_bytes(f"{subject}:{name}".encode())[:24],
                    "subject_id": subject,
                    "member_name": name,
                    "source_sha256": sha256_bytes(raw),
                    "normalized_signal_sha256": sha256_bytes(windows.tobytes()),
                    "label": label,
                    "windows": windows,
                }
            )
    return records


def _lstm_logits(torch, records):
    from torch import nn

    ckpt = torch.load(LSTM_CHECKPOINT, map_location="cpu", weights_only=True)
    if ckpt.get("class_ids") != list(ACTIVITY_IDS):
        raise ValueError("LSTM class order changed")
    lstm, classifier = nn.LSTM(3, 64, batch_first=True), nn.Linear(64, len(ACTIVITY_IDS))
    state = ckpt["state_dict"]
    lstm.load_state_dict(
        {
            key.removeprefix("lstm."): value
            for key, value in state.items()
            if key.startswith("lstm.")
        }
    )
    classifier.load_state_dict(
        {
            key.removeprefix("classifier."): value
            for key, value in state.items()
            if key.startswith("classifier.")
        }
    )
    device = torch.device("cuda")
    lstm, classifier = lstm.to(device).eval(), classifier.to(device).eval()
    mean, scale = ckpt["axis_mean"].to(device), ckpt["axis_scale"].to(device)
    result = []
    with torch.inference_mode():
        for record in records:
            windows = record["windows"]
            rows = []
            for start in range(0, len(windows), 256):
                batch = torch.as_tensor(
                    windows[start : start + 256], dtype=torch.float32, device=device
                )
                _, (hidden, _) = lstm(((batch - mean) / scale).transpose(1, 2).contiguous())
                rows.append(classifier(hidden[-1]).float().cpu().numpy())
            result.append(np.concatenate(rows).mean(axis=0))
    return np.stack(result).astype(np.float32)


def _moment_logits(torch, records):
    import sys

    commit = subprocess.run(
        ["git", "-C", str(MOMENT_SOURCE), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if (
        commit != MOMENT_SOURCE_COMMIT
        or sha256_file(MOMENT_DIR / "model.safetensors") != MOMENT_WEIGHTS_SHA256
    ):
        raise ValueError("MOMENT source or weights differ from the frozen revision")
    sys.path.insert(0, str(MOMENT_SOURCE.resolve()))
    from momentfm import MOMENTPipeline

    ckpt = torch.load(MOMENT_HEAD, map_location="cpu", weights_only=True)
    if ckpt.get("class_ids") != list(ACTIVITY_IDS) or ckpt.get("condition") != "moment_frozen":
        raise ValueError("MOMENT head differs from the frozen baseline")
    device = torch.device("cuda")
    encoder = MOMENTPipeline.from_pretrained(
        str(MOMENT_DIR.resolve()), model_kwargs={"task_name": "embedding"}, local_files_only=True
    ).to(device)
    encoder.init()
    encoder.eval()
    encoder.requires_grad_(False)
    head = torch.nn.Linear(512, len(ACTIVITY_IDS)).to(device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    mean, scale = ckpt["mean"].to(device), ckpt["scale"].to(device)
    result = []
    with torch.inference_mode():
        for record in records:
            windows = record["windows"]
            embeddings = []
            for start in range(0, len(windows), 64):
                batch = torch.as_tensor(
                    windows[start : start + 64], dtype=torch.float32, device=device
                )
                mask = torch.ones((len(batch), windows.shape[-1]), dtype=torch.bool, device=device)
                values = encoder(x_enc=batch, input_mask=mask, reduction="mean").embeddings
                if values.shape != (len(batch), 512) or not torch.isfinite(values).all():
                    raise RuntimeError("invalid MOMENT holdout embeddings")
                embeddings.append(values.float())
            features = torch.cat(embeddings).mean(dim=0, keepdim=True)
            result.append(head((features - mean) / scale).float().cpu().numpy()[0])
    del encoder
    return np.stack(result).astype(np.float32)


def run_evaluation(output_dir: Path, *, preflight_sha256: str) -> dict[str, Any]:
    if not output_dir.is_dir() or {path.name for path in output_dir.iterdir()} != {
        "preflight.json"
    }:
        raise ValueError("output must contain only its frozen preflight")
    preflight_path = output_dir / "preflight.json"
    if sha256_file(preflight_path) != preflight_sha256:
        raise ValueError("frozen preflight hash mismatch")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        preflight.get("script_sha256") != sha256_file(Path(__file__).resolve())
        or preflight.get("model_pins") != _pins()
    ):
        raise ValueError("script or frozen model artifacts changed after preregistration")
    from scripts.pch_moment_sensor_probe import gpu_preflight

    gate = gpu_preflight(dict(os.environ))
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable after HotUGPU preflight")
    torch.cuda.reset_peak_memory_stats()
    started, timer = datetime.now(UTC).isoformat(), time.perf_counter()
    records = _heldout_records()
    actual_subjects = sorted({row["subject_id"] for row in records})
    if len(records) != 84 or _names_hash(actual_subjects) != HELDOUT_SUBJECT_SHA256:
        raise ValueError("runtime holdout differs from the preregistered participant set")
    if (
        _names_hash([row["member_name"] for row in records])
        != preflight["heldout_member_names_sha256"]
    ):
        raise ValueError("runtime holdout recordings differ from preregistered member set")
    heldout_content = [row["source_sha256"] for row in records]
    heldout_signals = [row["normalized_signal_sha256"] for row in records]
    previous_content = list(set().union(*preflight["nonheldout_content_hashes"].values()))
    previous_signals = list(set().union(*preflight["nonheldout_normalized_signal_hashes"].values()))
    verify_content_separation(heldout_content, heldout_signals, previous_content, previous_signals)
    labels = {name: index for index, name in enumerate(ACTIVITY_IDS)}
    targets = np.asarray([labels[row["label"]] for row in records], dtype=np.int64)
    subjects = np.asarray([row["subject_id"] for row in records])
    logits_lstm, logits_moment = _lstm_logits(torch, records), _moment_logits(torch, records)
    metrics = {
        "window_lstm": summarize_logits(logits_lstm, targets, class_count=len(ACTIVITY_IDS)),
        "frozen_moment_plus_linear": summarize_logits(
            logits_moment, targets, class_count=len(ACTIVITY_IDS)
        ),
    }
    bootstrap = paired_subject_bootstrap(logits_lstm, logits_moment, targets, subjects)
    rows = []
    for record, target, lstm, moment in zip(
        records, targets, logits_lstm, logits_moment, strict=True
    ):
        rows.append(
            {
                "sample_id": record["sample_id"],
                "participant_id": record["subject_id"],
                "source_member_name_sha256": sha256_bytes(record["member_name"].encode()),
                "target_id": int(target),
                "target_name": record["label"],
                "lstm_prediction_id": int(lstm.argmax()),
                "lstm_probabilities": _probabilities(lstm[None])[0].tolist(),
                "moment_prediction_id": int(moment.argmax()),
                "moment_probabilities": _probabilities(moment[None])[0].tolist(),
            }
        )
    (output_dir / "heldout-predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    torch.cuda.synchronize()
    result = {
        "status": "PASS_PCH_HOLDOUT_EVALUATED",
        "purpose": "one-time inference-only participant holdout comparison",
        "started_at_utc": started,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "source_commit": preflight["source_commit"],
        "script_sha256": preflight["script_sha256"],
        "preflight_sha256": sha256_file(preflight_path),
        "archive_sha256": ARCHIVE_SHA256,
        "model_pins": preflight["model_pins"],
        "heldout_record_count": len(records),
        "heldout_subject_count": len(actual_subjects),
        "heldout_subject_ids_sha256": _names_hash(actual_subjects),
        "label_counts": dict(Counter(row["label"] for row in records)),
        "heldout_payload_opened": True,
        "training": False,
        "checkpoint_selection": False,
        "sealed_audit_loaded": False,
        "metrics": metrics,
        "paired_subject_bootstrap": bootstrap,
        "compute": {
            "device": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "elapsed_seconds_total": time.perf_counter() - timer,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "hotgpu_preflight": gate,
            "cloud_cost_usd": 0,
        },
        "interpretation_limit": "one optional sensor task on one fixed 28-participant holdout",
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output-dir", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--preflight-sha256", required=True)
    args = parser.parse_args()
    result = (
        prepare_preflight(args.output_dir)
        if args.command == "prepare"
        else run_evaluation(args.output_dir, preflight_sha256=args.preflight_sha256)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
