"""Fit a frozen MOMENT sensor representation on the public PCH Data 2017 task.

The script reads only fit and development-validation participants. The
participant-held-out group is assigned from anonymized IDs and never opened.
The decision target is one of the 18 supplied activity IDs; no text encoder or
question text is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ARCHIVE_SIZE = 98_272_925
ARCHIVE_SHA256 = "7814f871c834725d0108192cc9265b1abd2db8ead96815d82c828202f1e55ba9"
MOMENT_SOURCE_COMMIT = "38f7310ad594100747ca2a8357e9c7ca7d323e0e"
MOMENT_MODEL_REVISION = "411e288267f82cce86296dbe4d6c8bc533cc162f"
MOMENT_WEIGHTS_SHA256 = "785e6c6f57ffa7cac7e2a1fff6369618d49f2f441563ddea76c866231e5aa877"
SPLIT_NAMESPACE = "tiny-omni-pch2017-subject-split-v1"
SEED = 17
WINDOW_SAMPLES = 128
SAMPLE_RATE_HZ = 51.2

ACTIVITY_IDS = (
    "arm_curl",
    "bounce",
    "brushing_teeth",
    "catch",
    "chopping_vegetables",
    "deadlift",
    "hair_grooming",
    "ironing",
    "jogging",
    "lateral_arm_raise",
    "mixing_food",
    "pass",
    "sieving_flour",
    "stepping",
    "washing_dishes",
    "washing_hands",
    "washing_windows",
    "walking",
)

_ACTIVITY_ALIASES = {
    "arm_curl": ("armcurls", "armcurl", "arm_curls", "arm_curl"),
    "bounce": ("bounce", "bouncing"),
    "brushing_teeth": ("brushingteeth", "teethbrushing", "brushteeth"),
    "catch": ("catch",),
    "chopping_vegetables": (
        "choppingvegetables",
        "choppingvegetable",
        "choppingvegtables",
        "choppingveg",
        "chopveg",
        "chopping",
    ),
    "deadlift": ("deadlift", "deadlifts"),
    "hair_grooming": ("hairgrooming", "brushinghair", "hair"),
    "ironing": ("ironing",),
    "jogging": ("jogging", "running"),
    "lateral_arm_raise": (
        "lateralarmraises",
        "lateralarmraise",
        "lateralarms",
        "lateralraise",
        "lateralraises",
    ),
    "mixing_food": (
        "mixingfoodinabowl",
        "mixingfoodinbowl",
        "mixingfood",
        "mixingbowl",
        "bowlmix",
        "mixing",
    ),
    "pass": ("pass",),
    "sieving_flour": ("sievingflour", "seivingflour", "siev", "seiv"),
    "stepping": ("stepping",),
    "washing_dishes": (
        "washingdishes",
        "dishwashing",
        "cleandishes",
        "cleaningdishes",
        "washdish",
    ),
    "washing_hands": ("washinghands", "handwashing"),
    "washing_windows": (
        "washingwindows",
        "windowwashing",
        "windowcleaning",
        "windowncleaning",
        "washwindow",
    ),
    "walking": ("walking",),
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def subject_id_from_member(name: str) -> str:
    filename = Path(name).name
    match = re.match(r"(P\d{3})", filename, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"unrecognized anonymized participant ID in {filename!r}")
    return match.group(1).upper()


def assign_subject_splits(
    member_names: list[str], *, seed: int = SEED
) -> tuple[dict[str, str], dict[str, str]]:
    """Assign participants by immutable hash order without reading activity labels."""
    subjects = sorted({subject_id_from_member(name) for name in member_names})
    if len(subjects) < 5:
        raise ValueError("at least five participant groups are required")
    order = sorted(
        subjects,
        key=lambda subject: hashlib.sha256(f"{SPLIT_NAMESPACE}:{seed}:{subject}".encode()).digest(),
    )
    fit_count = round(0.65 * len(order))
    validation_count = round(0.15 * len(order))
    if fit_count < 1 or validation_count < 1 or fit_count + validation_count >= len(order):
        raise ValueError("split proportions leave an empty partition")
    split_by_subject = {
        subject: (
            "fit"
            if index < fit_count
            else "validation"
            if index < fit_count + validation_count
            else "heldout"
        )
        for index, subject in enumerate(order)
    }
    hashes = {
        split: sha256_bytes(
            "\n".join(sorted(s for s, value in split_by_subject.items() if value == split)).encode()
        )
        for split in ("fit", "validation", "heldout")
    }
    return split_by_subject, hashes


def canonical_activity(member_name: str) -> str:
    """Map filename variants to the source's 18 fixed action IDs."""
    normalized = re.sub(r"[^a-z0-9]", "", Path(member_name).stem.casefold())
    matches = []
    for activity, aliases in _ACTIVITY_ALIASES.items():
        for alias in aliases:
            token = re.sub(r"[^a-z0-9]", "", alias.casefold())
            if token in normalized:
                matches.append((len(token), activity))
    if not matches:
        raise ValueError(f"unmapped activity in TRAIN/validation filename: {member_name!r}")
    longest = max(length for length, _ in matches)
    selected = {activity for length, activity in matches if length == longest}
    if len(selected) != 1:
        raise ValueError(f"ambiguous activity aliases in {member_name!r}: {sorted(selected)}")
    return selected.pop()


def stratify_validation_with_fit_only_labels(
    member_names: list[str], split_by_subject: dict[str, str], *, seed: int = SEED
) -> dict[str, str]:
    """Swap fit participants into validation until every class is represented.

    Held-out filenames are excluded before activity parsing. The held-out group
    remains the original hash assignment; only fit/validation participants are
    eligible for deterministic coverage swaps.
    """
    labels_by_subject: dict[str, set[str]] = defaultdict(set)
    for member in member_names:
        subject = subject_id_from_member(member)
        if split_by_subject[subject] != "heldout":
            labels_by_subject[subject].add(canonical_activity(member))

    splits = dict(split_by_subject)
    min_validation_subjects_per_class = 2
    while True:
        validation_subject_counts = Counter(
            label
            for subject, label_set in labels_by_subject.items()
            if splits[subject] == "validation"
            for label in label_set
        )
        current_deficit = sum(
            max(0, min_validation_subjects_per_class - validation_subjects)
            for validation_subjects in (validation_subject_counts[label] for label in ACTIVITY_IDS)
        )
        if current_deficit == 0:
            break
        swaps = []
        fit_subjects = [s for s, value in splits.items() if value == "fit"]
        validation_subjects = [s for s, value in splits.items() if value == "validation"]
        for candidate in fit_subjects:
            for removed in validation_subjects:
                next_counts = validation_subject_counts.copy()
                next_counts.subtract(labels_by_subject[removed])
                next_counts.update(labels_by_subject[candidate])
                next_deficit = sum(
                    max(0, min_validation_subjects_per_class - next_counts[label])
                    for label in ACTIVITY_IDS
                )
                improvement = current_deficit - next_deficit
                if improvement > 0:
                    swaps.append((improvement, candidate, removed))
        if not swaps:
            missing = {
                label: validation_subject_counts[label]
                for label in ACTIVITY_IDS
                if validation_subject_counts[label] < min_validation_subjects_per_class
            }
            raise ValueError(
                f"cannot achieve validation coverage without held-out participants: {missing}"
            )
        improvement, candidate, removed = min(
            swaps,
            key=lambda swap: (
                -swap[0],
                hashlib.sha256(f"{SPLIT_NAMESPACE}:{seed}:coverage:{swap[1]}".encode()).digest(),
                hashlib.sha256(f"{SPLIT_NAMESPACE}:{seed}:swap-out:{swap[2]}".encode()).digest(),
            ),
        )
        del improvement
        splits[candidate] = "validation"
        splits[removed] = "fit"

    if sum(value == "validation" for value in splits.values()) != sum(
        value == "validation" for value in split_by_subject.values()
    ):
        raise AssertionError("validation coverage swaps changed the participant count")
    return splits


def decode_recording(raw: bytes) -> tuple[np.ndarray, np.ndarray, int]:
    """Read PCH's calibrated timestamp and calibrated triaxial acceleration."""
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"))))
    if len(rows) <= 4 or len(rows[1]) < 8:
        raise ValueError("PCH CSV does not match the verified 4-header, 8-column layout")
    expected_columns = [
        "Timestamp",
        "Accelerometer X",
        "Accelerometer Y",
        "Accelerometer Z",
    ]
    if [value.strip() for value in rows[1][4:8]] != expected_columns:
        raise ValueError("calibrated PCH columns changed")
    extra_headers = [value.strip() for value in rows[1][8:]]
    if any(value != "Gyroscope X" for value in extra_headers):
        raise ValueError("PCH CSV contains unexpected non-calibrated extra channels")
    values = []
    for row in rows[4:]:
        if len(row) < 7:
            raise ValueError("PCH row omits the timestamp or two calibrated axes")
        calibrated = row[4:8] + [""] * max(0, 8 - len(row))
        parsed = [float(value) if value.strip() else np.nan for value in calibrated]
        values.append(parsed)
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < WINDOW_SAMPLES:
        raise ValueError("PCH calibrated data must contain a full window")
    timestamps_ms, axes = array[:, 0], array[:, 1:].astype(np.float32)
    if not np.isfinite(timestamps_ms).all():
        raise ValueError("PCH timestamps cannot be missing or nonfinite")
    segments = timestamp_segments(timestamps_ms)
    interpolated_count = 0
    for channel in range(axes.shape[1]):
        valid = np.isfinite(axes[:, channel])
        missing_count = int((~valid).sum())
        if missing_count / len(axes) > 0.01 or valid.sum() < WINDOW_SAMPLES:
            raise ValueError("PCH axis missingness exceeds the 1% interpolation limit")
        if missing_count:
            for segment in segments:
                segment_valid = np.isfinite(axes[segment, channel])
                if segment_valid.all():
                    continue
                segment_times = timestamps_ms[segment]
                segment_axes = axes[segment, channel]
                if segment_valid.sum() < 2:
                    raise ValueError("cannot interpolate missing axis values within a time segment")
                segment_axes[~segment_valid] = np.interp(
                    segment_times[~segment_valid],
                    segment_times[segment_valid],
                    segment_axes[segment_valid],
                )
            interpolated_count += missing_count
    deltas = np.diff(timestamps_ms)
    positive_deltas = deltas[deltas > 0]
    if not len(positive_deltas):
        raise ValueError("calibrated PCH timestamps contain no positive intervals")
    median_delta = float(np.median(positive_deltas))
    expected_delta = 1000.0 / SAMPLE_RATE_HZ
    if not math.isclose(median_delta, expected_delta, rel_tol=0.01, abs_tol=0.5):
        raise ValueError("calibrated timestamp rate differs from the declared 51.2 Hz")
    return timestamps_ms, axes, interpolated_count


def timestamp_segments(timestamps_ms: np.ndarray) -> list[slice]:
    if timestamps_ms.ndim != 1 or len(timestamps_ms) < 2:
        raise ValueError("at least two timestamps are required")
    deltas = np.diff(timestamps_ms)
    positive_deltas = deltas[deltas > 0]
    if not len(positive_deltas):
        raise ValueError("timestamps have no positive intervals")
    median_delta = float(np.median(positive_deltas))
    breaks = np.flatnonzero((deltas <= 0) | (deltas > 3.0 * median_delta)) + 1
    boundaries = np.concatenate(([0], breaks, [len(timestamps_ms)]))
    return [
        slice(int(start), int(end))
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]


def resample_and_window(
    timestamps_ms: np.ndarray,
    axes: np.ndarray,
    *,
    window_samples: int = WINDOW_SAMPLES,
) -> np.ndarray:
    """Linearly resample to 51.2 Hz and return nonoverlapping [N,3,128] windows."""
    if axes.ndim != 2 or axes.shape[1] != 3 or len(timestamps_ms) != len(axes):
        raise ValueError("accelerometer input must align as [time, 3]")
    if window_samples < 1 or not np.isfinite(axes).all():
        raise ValueError("window size must be positive and axes finite")
    segments = []
    for segment in timestamp_segments(timestamps_ms):
        segment_times = timestamps_ms[segment]
        segment_axes = axes[segment]
        if len(segment_times) < 2:
            continue
        start, end = float(segment_times[0]), float(segment_times[-1])
        count = int(math.floor((end - start) / (1000.0 / SAMPLE_RATE_HZ))) + 1
        if count < window_samples:
            continue
        grid = start + np.arange(count, dtype=np.float64) * (1000.0 / SAMPLE_RATE_HZ)
        resampled = np.stack(
            [np.interp(grid, segment_times, segment_axes[:, channel]) for channel in range(3)],
            axis=1,
        ).astype(np.float32)
        usable = (len(resampled) // window_samples) * window_samples
        segments.append(resampled[:usable].reshape(-1, window_samples, 3).transpose(0, 2, 1))
    if not segments:
        raise ValueError("recording contains no complete window inside a continuous segment")
    windows = np.concatenate(segments, axis=0)
    if windows.size == 0 or not np.isfinite(windows).all():
        raise ValueError("windowing produced no finite sensor observations")
    return windows.copy()


def handcrafted_features(windows: np.ndarray) -> np.ndarray:
    """Compute fixed, text-free statistical features for the non-pretrained control."""
    if windows.ndim != 3 or windows.shape[1:] != (3, WINDOW_SAMPLES):
        raise ValueError("expected sensor windows with shape [N, 3, 128]")
    values = windows.transpose(0, 2, 1).astype(np.float64)
    magnitude = np.linalg.norm(values, axis=-1, keepdims=True)
    joined = np.concatenate((values, magnitude), axis=-1)
    quantiles = np.quantile(joined, [0.05, 0.25, 0.5, 0.75, 0.95], axis=1)
    features = np.concatenate(
        (
            joined.mean(axis=1),
            joined.std(axis=1),
            joined.min(axis=1),
            joined.max(axis=1),
            np.sqrt(np.mean(joined**2, axis=1)),
            quantiles.transpose(1, 0, 2).reshape(len(windows), -1),
        ),
        axis=1,
    )
    return features.astype(np.float32)


def gpu_preflight(environment: dict[str, str]) -> dict[str, Any]:
    """Check an idle GPU or an exact cooperative HotUGPU reservation before CUDA import."""
    query_pids = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        check=True,
        text=True,
    )
    active_pids = {
        int(line.strip())
        for line in query_pids.stdout.splitlines()
        if line.strip() and line.strip().casefold() != "no running processes found"
    }
    telemetry = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,name",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    fields = [part.strip() for part in telemetry.stdout.splitlines()[0].split(",")]
    if len(fields) != 4:
        raise RuntimeError("unexpected nvidia-smi telemetry shape")
    repository_root = str(Path(__file__).resolve().parents[1])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)
    from scripts.hotgpu_preflight import validate_gpu_preflight

    report = validate_gpu_preflight(
        memory_used_mib=int(fields[1]),
        utilization_pct=int(fields[0]),
        total_mib=int(fields[2]),
        active_pids=active_pids,
        own_pid=os.getpid(),
        environment=environment,
    )
    return {**report, "gpu_name": fields[3], "active_pids": sorted(active_pids)}


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_records(archive: zipfile.ZipFile) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    members = [
        info.filename
        for info in archive.infolist()
        if not info.is_dir() and info.filename.casefold().endswith(".csv")
    ]
    split_by_subject, _ = assign_subject_splits(members)
    split_by_subject = stratify_validation_with_fit_only_labels(members, split_by_subject)
    subject_hashes = {
        split: sha256_bytes(
            "\n".join(sorted(s for s, value in split_by_subject.items() if value == split)).encode()
        )
        for split in ("fit", "validation", "heldout")
    }
    split_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for member in sorted(members):
        subject = subject_id_from_member(member)
        split = split_by_subject[subject]
        if split == "heldout":
            # Do not parse the filename label or open the held-out member.
            continue
        label = canonical_activity(member)
        info = archive.getinfo(member)
        raw = archive.read(member)
        content_hash = sha256_bytes(raw)
        timestamps, axes, interpolated_count = decode_recording(raw)
        windows = resample_and_window(timestamps, axes)
        split_records[split].append(
            {
                "sample_id": sha256_bytes(f"{subject}:{member}".encode())[:24],
                "subject_id": subject,
                "source_member": member,
                "source_size": info.file_size,
                "source_crc32": f"{info.CRC:08x}",
                "source_sha256": content_hash,
                "normalized_signal_sha256": sha256_bytes(windows.tobytes()),
                "label": label,
                "windows": windows,
                "static_features": handcrafted_features(windows).mean(axis=0),
                "window_count": len(windows),
                "sensor_segment_count": len(timestamp_segments(timestamps)),
                "interpolated_axis_values": interpolated_count,
                "median_interval_ms": float(np.median(np.diff(timestamps))),
            }
        )
    fit_records = split_records["fit"]
    validation_records = split_records["validation"]
    fit_content = {r["source_sha256"] for r in fit_records}
    validation_content = {r["source_sha256"] for r in validation_records}
    if fit_content & validation_content:
        raise ValueError("exact recording-content overlap across fit/validation")
    fit_signals = {r["normalized_signal_sha256"] for r in fit_records}
    validation_signals = {r["normalized_signal_sha256"] for r in validation_records}
    if fit_signals & validation_signals:
        raise ValueError("normalized sensor-signal overlap across fit/validation")
    if {r["subject_id"] for r in fit_records} & {r["subject_id"] for r in validation_records}:
        raise ValueError("participant overlap across fit/validation")
    records = fit_records + validation_records
    metadata = {
        "expected_source_csv_count": 423,
        "observed_source_csv_count": len(members),
        "observed_subject_count": len(split_by_subject),
        "subject_counts": dict(Counter(split_by_subject.values())),
        "record_counts": {
            "fit": len(fit_records),
            "validation": len(validation_records),
            "heldout_not_opened": sum(value == "heldout" for value in split_by_subject.values()),
        },
        "interpolated_axis_values_fit_validation": sum(
            int(record["interpolated_axis_values"]) for record in fit_records + validation_records
        ),
        "subject_id_set_sha256_by_split": subject_hashes,
        "split_method": (
            "seed-17 participant hash split with fit-only deterministic swaps to represent "
            "every activity in validation; heldout assignment unchanged"
        ),
        "classes": list(ACTIVITY_IDS),
        "fit_class_counts": dict(Counter(r["label"] for r in fit_records)),
        "validation_class_counts": dict(Counter(r["label"] for r in validation_records)),
        "fit_subjects": sorted({r["subject_id"] for r in fit_records}),
        "validation_subjects": sorted({r["subject_id"] for r in validation_records}),
        "heldout_subject_ids_sha256": subject_hashes["heldout"],
        "heldout_labels_or_samples_loaded": False,
        "window_samples": WINDOW_SAMPLES,
        "window_stride_samples": WINDOW_SAMPLES,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "timestamp_column": "calibrated Timestamp (mSecs)",
        "sensor_columns": [
            "calibrated Accelerometer X (m/(sec^2))",
            "calibrated Accelerometer Y (m/(sec^2))",
            "calibrated Accelerometer Z (m/(sec^2))",
        ],
        "resampling": (
            "linear interpolation on calibrated timestamps at 51.2 Hz; "
            "drop incomplete trailing window; split at gaps >3x median interval; "
            "interpolate sparse missing axis values only when <=1% per axis"
        ),
        "no_text_input": True,
    }
    if len(members) != 423 or len(split_by_subject) != 141:
        raise ValueError("pinned PCH release inventory differs from 423 recordings / 141 subjects")
    return records, metadata


def preflight_inventory(
    records: list[dict[str, Any]], metadata: dict[str, Any], *, batch_size: int
) -> dict[str, Any]:
    fit = [record for record in records if subject_split_for_record(record, metadata) == "fit"]
    validation = [
        record for record in records if subject_split_for_record(record, metadata) == "validation"
    ]

    def class_subject_counts(selected: list[dict[str, Any]]) -> dict[str, int]:
        return {
            activity: len(
                {record["subject_id"] for record in selected if record["label"] == activity}
            )
            for activity in ACTIVITY_IDS
        }

    return {
        "fit_sample_id_order_sha256": sha256_bytes(
            "\n".join(record["sample_id"] for record in fit).encode()
        ),
        "validation_sample_id_order_sha256": sha256_bytes(
            "\n".join(record["sample_id"] for record in validation).encode()
        ),
        "record_counts": metadata["record_counts"],
        "sample_counts_by_class": {
            "fit": metadata["fit_class_counts"],
            "validation": metadata["validation_class_counts"],
        },
        "unique_subject_counts_by_split": {
            "fit": len(metadata["fit_subjects"]),
            "validation": len(metadata["validation_subjects"]),
            "heldout": metadata["subject_counts"]["heldout"],
        },
        "class_subject_counts": {
            "fit": class_subject_counts(fit),
            "validation": class_subject_counts(validation),
        },
        "windows_by_split": {
            "fit": sum(record["window_count"] for record in fit),
            "validation": sum(record["window_count"] for record in validation),
        },
        "total_windows": sum(record["window_count"] for record in records),
        "inference_batches": math.ceil(
            sum(record["window_count"] for record in records) / batch_size
        ),
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "window_samples": WINDOW_SAMPLES,
        "window_stride_samples": WINDOW_SAMPLES,
        "batch_size": batch_size,
        "classes": list(ACTIVITY_IDS),
        "split_method": metadata["split_method"],
        "subject_id_set_sha256_by_split": metadata["subject_id_set_sha256_by_split"],
        "heldout_labels_or_samples_loaded": False,
        "raw_content_overlap": False,
        "normalized_signal_overlap": False,
    }


def make_preflight(
    *, archive_path: Path, model_dir: Path, moment_source: Path, batch_size: int
) -> dict[str, Any]:
    if (
        Path(archive_path).stat().st_size != ARCHIVE_SIZE
        or sha256_file(archive_path) != ARCHIVE_SHA256
    ):
        raise ValueError("PCH archive size/SHA-256 differs from the acquired source")
    weights_path = Path(model_dir) / "model.safetensors"
    if sha256_file(weights_path) != MOMENT_WEIGHTS_SHA256:
        raise ValueError("MOMENT weights differ from the pinned R1 checkpoint")
    source_commit = subprocess.run(
        ["git", "-C", str(moment_source), "rev-parse", "HEAD"],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    if source_commit != MOMENT_SOURCE_COMMIT:
        raise ValueError("MOMENT source checkout differs from the pinned R1 audit")
    with zipfile.ZipFile(archive_path) as archive:
        records, metadata = build_records(archive)
    return {
        "status": "PREFLIGHT_PASS",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "archive_sha256": ARCHIVE_SHA256,
        "archive_bytes": ARCHIVE_SIZE,
        "model_revision": MOMENT_MODEL_REVISION,
        "model_weights_sha256": MOMENT_WEIGHTS_SHA256,
        "moment_source_commit": source_commit,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "seed": SEED,
        "split_metadata": metadata,
        "inventory": preflight_inventory(records, metadata, batch_size=batch_size),
    }


def _metric_summary(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if (
        probabilities.ndim != 2
        or probabilities.shape[0] != len(targets)
        or probabilities.shape[1] != len(ACTIVITY_IDS)
        or not np.isfinite(probabilities).all()
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise ValueError("expected finite normalized probabilities for every fixed activity")
    predictions = probabilities.argmax(axis=1)
    correct = predictions == targets
    clipped = np.clip(probabilities, np.finfo(np.float64).tiny, 1.0)
    nll = -np.log(clipped[np.arange(len(targets)), targets])
    one_hot = np.eye(len(ACTIVITY_IDS))[targets]
    confidence = probabilities.max(axis=1)
    ece = 0.0
    for index in range(15):
        lower, upper = index / 15, (index + 1) / 15
        mask = (confidence >= lower) & (confidence < upper if index < 14 else confidence <= upper)
        if mask.any():
            ece += float(mask.mean()) * abs(float(confidence[mask].mean() - correct[mask].mean()))
    class_accuracy = {}
    class_count = {}
    for class_index, activity in enumerate(ACTIVITY_IDS):
        mask = targets == class_index
        class_count[activity] = int(mask.sum())
        class_accuracy[activity] = float(correct[mask].mean()) if mask.any() else None
    present = [value for value in class_accuracy.values() if value is not None]
    return {
        "n_recordings": len(targets),
        "accuracy": float(correct.mean()),
        "macro_accuracy_present_classes": float(np.mean(present)) if present else None,
        "nll": float(nll.mean()),
        "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ece_15": ece,
        "mean_confidence": float(confidence.mean()),
        "class_counts": class_count,
        "class_accuracy": class_accuracy,
    }


def participant_bootstrap_comparison(
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    targets: np.ndarray,
    subject_ids: list[str],
    *,
    seed: int = SEED,
    replicates: int = 2000,
) -> dict[str, Any]:
    """Paired participant-cluster bootstrap for MOMENT minus static control."""
    if len(subject_ids) != len(targets) or replicates < 100:
        raise ValueError("bootstrap requires aligned records and at least 100 replicates")
    subjects = np.asarray(sorted(set(subject_ids)))
    subject_array = np.asarray(subject_ids)
    if len(subjects) < 2:
        raise ValueError("participant-cluster bootstrap needs multiple validation subjects")
    rng = np.random.default_rng(seed)
    moment_accuracy, static_accuracy, delta_accuracy = [], [], []
    moment_nll, static_nll, delta_nll = [], [], []
    for _ in range(replicates):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        indices = np.concatenate([np.flatnonzero(subject_array == subject) for subject in sampled])
        target = targets[indices]
        selected_a, selected_b = predictions_a[indices], predictions_b[indices]
        accuracy_a = float((selected_a.argmax(axis=1) == target).mean())
        accuracy_b = float((selected_b.argmax(axis=1) == target).mean())
        nll_a = float(
            -np.log(np.clip(selected_a[np.arange(len(target)), target], 1e-12, 1.0)).mean()
        )
        nll_b = float(
            -np.log(np.clip(selected_b[np.arange(len(target)), target], 1e-12, 1.0)).mean()
        )
        moment_accuracy.append(accuracy_a)
        static_accuracy.append(accuracy_b)
        delta_accuracy.append(accuracy_a - accuracy_b)
        moment_nll.append(nll_a)
        static_nll.append(nll_b)
        delta_nll.append(nll_a - nll_b)

    def interval(values: list[float]) -> list[float]:
        return [float(value) for value in np.quantile(values, [0.025, 0.975])]

    return {
        "method": "paired participant-cluster bootstrap, percentile 95% interval",
        "seed": seed,
        "replicates": replicates,
        "validation_subjects": int(len(subjects)),
        "delta_direction": "moment_frozen minus static_statistics",
        "delta_accuracy_95_percent_interval": interval(delta_accuracy),
        "delta_nll_95_percent_interval": interval(delta_nll),
        "moment_accuracy_95_percent_interval": interval(moment_accuracy),
        "moment_nll_95_percent_interval": interval(moment_nll),
        "static_accuracy_95_percent_interval": interval(static_accuracy),
        "static_nll_95_percent_interval": interval(static_nll),
    }


def _train_linear_head(
    fit_features: np.ndarray,
    fit_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    *,
    seed: int,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    from torch import nn
    from torch.nn import functional as F

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    x_train = torch.as_tensor(fit_features, dtype=torch.float32, device=device)
    y_train = torch.as_tensor(fit_targets, dtype=torch.long, device=device)
    x_validation = torch.as_tensor(validation_features, dtype=torch.float32, device=device)
    y_validation = torch.as_tensor(validation_targets, dtype=torch.long, device=device)
    mean = x_train.mean(dim=0, keepdim=True)
    scale = x_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    x_train = (x_train - mean) / scale
    x_validation = (x_validation - mean) / scale
    model = nn.Linear(x_train.shape[1], len(ACTIVITY_IDS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    best_nll = float("inf")
    best_state = None
    best_epoch = 0
    best_metrics = None
    history = []
    patience = 25
    min_delta = 1e-5
    for epoch in range(1, 301):
        model.train()
        order = torch.randperm(len(x_train), device=device)
        epoch_loss_total = 0.0
        epoch_examples = 0
        for start in range(0, len(order), 64):
            batch = order[start : start + 64]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_train[batch])
            loss = F.cross_entropy(logits, y_train[batch])
            loss.backward()
            optimizer.step()
            epoch_loss_total += float(loss.detach().cpu()) * len(batch)
            epoch_examples += len(batch)
        model.eval()
        with torch.inference_mode():
            train_logits = model(x_train)
            train_ce = float(F.cross_entropy(train_logits, y_train))
            train_accuracy = float((train_logits.argmax(dim=-1) == y_train).float().mean())
            logits = model(x_validation)
            validation_nll = float(F.cross_entropy(logits, y_validation))
            probabilities = logits.softmax(dim=-1).cpu().numpy()
        metrics = _metric_summary(probabilities, validation_targets)
        history.append(
            {
                "epoch": epoch,
                "train_ce": epoch_loss_total / epoch_examples,
                "train_accuracy": train_accuracy,
                "train_nll": train_ce,
                "validation_nll": validation_nll,
                "validation_accuracy": metrics["accuracy"],
                "validation_brier": metrics["brier"],
                "validation_ece_15": metrics["ece_15"],
            }
        )
        if validation_nll < best_nll - min_delta:
            best_nll = validation_nll
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            best_metrics = metrics
        elif epoch - best_epoch >= patience:
            break
    if best_state is None or best_metrics is None:
        raise RuntimeError("linear head did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        validation_probabilities = model(x_validation).softmax(dim=-1).cpu().numpy()
    return (
        {
            "state_dict": best_state,
            "mean": mean.detach().cpu(),
            "scale": scale.detach().cpu(),
            "best_epoch": best_epoch,
            "best_validation_metrics": best_metrics,
            "history": history,
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "optimizer": "AdamW",
            "learning_rate": 0.01,
            "weight_decay": 0.01,
            "batch_size": 64,
            "max_epochs": 300,
            "early_stopping_patience": patience,
            "selection": "minimum validation NLL, min_delta=1e-5",
        },
        {
            "fit_probabilities": None,
            "validation_probabilities": validation_probabilities,
            "fit_targets": fit_targets,
            "validation_targets": validation_targets,
        },
    )


def run_probe(
    *,
    archive_path: Path,
    model_dir: Path,
    moment_source: Path,
    output_dir: Path,
    batch_size: int = 64,
) -> dict[str, Any]:
    run_started_at = datetime.now(UTC).isoformat()
    run_timer = time.perf_counter()
    preflight_path = output_dir / "preflight.json"
    if not preflight_path.is_file() or {path.name for path in output_dir.iterdir()} != {
        "preflight.json"
    }:
        raise ValueError("output directory must contain only this run's preflight.json")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        Path(archive_path).stat().st_size != ARCHIVE_SIZE
        or sha256_file(archive_path) != ARCHIVE_SHA256
    ):
        raise ValueError("PCH archive size/SHA-256 differs from the acquired source")
    weights_path = Path(model_dir) / "model.safetensors"
    if sha256_file(weights_path) != MOMENT_WEIGHTS_SHA256:
        raise ValueError("MOMENT weights differ from the pinned R1 checkpoint")
    source_commit = subprocess.run(
        ["git", "-C", str(moment_source), "rev-parse", "HEAD"],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    if source_commit != MOMENT_SOURCE_COMMIT:
        raise ValueError("MOMENT source checkout differs from the pinned R1 audit")
    if (
        preflight.get("archive_sha256") != ARCHIVE_SHA256
        or preflight.get("model_weights_sha256") != MOMENT_WEIGHTS_SHA256
        or preflight.get("moment_source_commit") != source_commit
        or preflight.get("script_sha256") != sha256_file(Path(__file__).resolve())
        or preflight.get("inventory", {}).get("batch_size") != batch_size
    ):
        raise ValueError("preflight does not match the current data/model/code/run settings")

    gate = gpu_preflight(dict(os.environ))
    sys.path.insert(0, str(Path(moment_source).resolve()))
    import torch
    from momentfm import MOMENTPipeline

    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    model = MOMENTPipeline.from_pretrained(
        str(Path(model_dir).resolve()),
        model_kwargs={"task_name": "embedding"},
        local_files_only=True,
    ).to(device)
    model.init()
    model.eval()
    model.requires_grad_(False)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    torch.cuda.reset_peak_memory_stats()

    preprocess_timer = time.perf_counter()
    with zipfile.ZipFile(archive_path) as archive:
        records, split_metadata = build_records(archive)
        preprocessing_seconds = time.perf_counter() - preprocess_timer
    current_inventory = preflight_inventory(records, split_metadata, batch_size=batch_size)
    if split_metadata != preflight["split_metadata"] or current_inventory != preflight["inventory"]:
        raise ValueError("runtime split/sample inventory differs from the saved preflight")
    window_refs = []
    for record_index, record in enumerate(records):
        for window in record["windows"]:
            window_refs.append((record_index, window))
    features_by_record = [[] for _ in records]
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(window_refs), batch_size):
            selected = window_refs[start : start + batch_size]
            batch = torch.from_numpy(np.stack([item[1] for item in selected])).to(device)
            mask = torch.ones((len(selected), WINDOW_SAMPLES), dtype=torch.bool, device=device)
            embeddings = model(x_enc=batch, input_mask=mask, reduction="mean").embeddings
            if embeddings.shape != (len(selected), 512) or not torch.isfinite(embeddings).all():
                raise RuntimeError("MOMENT returned an invalid sensor embedding")
            embeddings = embeddings.float().cpu().numpy()
            for (record_index, _), embedding in zip(selected, embeddings, strict=True):
                features_by_record[record_index].append(embedding)
    torch.cuda.synchronize()
    feature_seconds = time.perf_counter() - started
    for record, embedding_rows in zip(records, features_by_record, strict=True):
        if not embedding_rows:
            raise RuntimeError("recording has no MOMENT window embeddings")
        record["moment_feature"] = np.stack(embedding_rows).mean(axis=0).astype(np.float32)

    fit = [
        record for record in records if subject_split_for_record(record, split_metadata) == "fit"
    ]
    validation = [
        record
        for record in records
        if subject_split_for_record(record, split_metadata) == "validation"
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = {activity: index for index, activity in enumerate(ACTIVITY_IDS)}
    fit_targets = np.asarray([labels[record["label"]] for record in fit], dtype=np.int64)
    validation_targets = np.asarray(
        [labels[record["label"]] for record in validation], dtype=np.int64
    )
    feature_cache_path = output_dir / "development-features.npz"
    np.savez_compressed(
        feature_cache_path,
        fit_moment=np.stack([record["moment_feature"] for record in fit]),
        validation_moment=np.stack([record["moment_feature"] for record in validation]),
        fit_static=np.stack([record["static_features"] for record in fit]),
        validation_static=np.stack([record["static_features"] for record in validation]),
        fit_targets=fit_targets,
        validation_targets=validation_targets,
        fit_sample_ids=np.asarray([record["sample_id"] for record in fit]),
        validation_sample_ids=np.asarray([record["sample_id"] for record in validation]),
        fit_subject_ids=np.asarray([record["subject_id"] for record in fit]),
        validation_subject_ids=np.asarray([record["subject_id"] for record in validation]),
    )
    results = {}
    predictions_by_condition = {}
    head_timer = time.perf_counter()
    for name, key in (
        ("moment_frozen", "moment_feature"),
        ("static_statistics", "static_features"),
    ):
        train_x = np.stack([record[key] for record in fit])
        validation_x = np.stack([record[key] for record in validation])
        head, predictions = _train_linear_head(
            train_x,
            fit_targets,
            validation_x,
            validation_targets,
            seed=SEED,
            device="cpu",
        )
        validation_metrics = _metric_summary(
            predictions["validation_probabilities"], validation_targets
        )
        results[name] = {
            "validation_metrics": validation_metrics,
            "best_epoch": head["best_epoch"],
            "best_validation_nll": head["best_validation_metrics"]["nll"],
            "trainable_parameters": head["trainable_parameters"],
            "input_dimension": int(train_x.shape[1]),
            "selection": head["selection"],
            "history": head["history"],
        }
        predictions_by_condition[name] = predictions["validation_probabilities"]
        torch.save(
            {
                "state_dict": head["state_dict"],
                "mean": head["mean"],
                "scale": head["scale"],
                "class_ids": list(ACTIVITY_IDS),
                "selected_epoch": head["best_epoch"],
                "condition": name,
            },
            output_dir / f"{name}-best.pt",
        )
        (output_dir / f"{name}-history.json").write_text(
            json.dumps(head["history"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    head_training_seconds = time.perf_counter() - head_timer

    prediction_rows = []
    for row_index, record in enumerate(validation):
        prediction_rows.append(
            {
                "sample_id": record["sample_id"],
                "subject_id": record["subject_id"],
                "target_id": record["label"],
                "window_count": record["window_count"],
                "moment_probabilities": predictions_by_condition["moment_frozen"][
                    row_index
                ].tolist(),
                "static_probabilities": predictions_by_condition["static_statistics"][
                    row_index
                ].tolist(),
            }
        )
    (output_dir / "validation-predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )
    bootstrap = participant_bootstrap_comparison(
        predictions_by_condition["moment_frozen"],
        predictions_by_condition["static_statistics"],
        validation_targets,
        [record["subject_id"] for record in validation],
    )
    trainable_head = results["moment_frozen"]["trainable_parameters"]
    summary = {
        "status": "PASS_PCH_MOMENT_SENSOR_DEV_PROBE",
        "task": "direct text-free 18-way wearable activity decision per recording",
        "source": {
            "name": "PCH Data 2017",
            "doi": "10.21251/b9566edc-c68c-4eee-b67e-c0228d0d3833",
            "license": "CC BY 4.0",
            "archive_sha256": ARCHIVE_SHA256,
            "archive_bytes": ARCHIVE_SIZE,
        },
        "experiment": {
            "started_at_utc": run_started_at,
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - run_timer,
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "seed": SEED,
            "split_seed": SEED,
            "split_namespace": SPLIT_NAMESPACE,
            "window_samples": WINDOW_SAMPLES,
            "window_seconds": WINDOW_SAMPLES / SAMPLE_RATE_HZ,
            "resample_hz": SAMPLE_RATE_HZ,
            "batch_size": batch_size,
            "preprocessing_seconds": preprocessing_seconds,
            "feature_extraction_seconds": feature_seconds,
            "head_training_seconds": head_training_seconds,
            "validation_sample_id_order_sha256": sha256_bytes(
                "\n".join(record["sample_id"] for record in validation).encode()
            ),
            "preflight_sha256": sha256_file(preflight_path),
            "fit_sample_id_order_sha256": sha256_bytes(
                "\n".join(record["sample_id"] for record in fit).encode()
            ),
        },
        "split": split_metadata,
        "model": {
            "name": "AutonLab/MOMENT-1-small",
            "revision": MOMENT_MODEL_REVISION,
            "source_commit": source_commit,
            "weights_sha256": MOMENT_WEIGHTS_SHA256,
            "encoder_parameters": parameters,
            "encoder_trainable_parameters": 0,
            "trainable_head_parameters": trainable_head,
            "model_plus_head_parameters": parameters + trainable_head,
            "frozen": True,
            "parameter_dtype": str(next(model.parameters()).dtype),
            "embedding_reduction": "mean over 3 axes and 16 patches",
        },
        "baseline": "train-only standardized fixed statistics with same linear-head trainer",
        "metrics": results,
        "participant_bootstrap": bootstrap,
        "feature_cache": {
            "path": str(feature_cache_path),
            "sha256": sha256_file(feature_cache_path),
            "bytes": feature_cache_path.stat().st_size,
            "contains_fit_and_development_validation_only": True,
        },
        "compute": {
            "device": gate["gpu_name"],
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device_name": torch.cuda.get_device_name(device),
            "device_capability": list(torch.cuda.get_device_capability(device)),
            "hotgpu_preflight": gate,
            "feature_extraction_seconds": feature_seconds,
            "feature_windows": len(window_refs),
            "batch_size": batch_size,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "head_training_device": "cpu",
            "cloud_cost_usd": 0,
            "encoder_optimizer_updates": 0,
        },
        "validation_only_model_selection": True,
        "heldout_opened": False,
        "quality_scope": "single narrow task, one fixed subject split, development validation only",
        "warnings": [
            (
                "PCH was published in 2026 after the MOMENT release, but the dataset was "
                "described in a 2018 paper; exact overlap with MOMENT pretraining was not verified."
            ),
            "The 18-class fixed taxonomy is not general sensor reasoning or full Omni quality.",
            "Validation contains 21 participants; uncertainty must be clustered by participant.",
            (
                "All six food-preparation validation recordings come from two participants; "
                "class-specific estimates are low precision."
            ),
        ],
    }
    _atomic_json(output_dir / "result.json", summary)
    return summary


def subject_split_for_record(record: dict[str, Any], metadata: dict[str, Any]) -> str:
    subject = record["subject_id"]
    if subject in metadata["fit_subjects"]:
        return "fit"
    if subject in metadata["validation_subjects"]:
        return "validation"
    raise ValueError("record references an unknown or held-out subject")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--moment-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 256:
        raise ValueError("batch size must be between 1 and 256")
    if args.preflight_only:
        if args.output_dir.exists():
            raise ValueError("preflight output directory must be new")
        args.output_dir.mkdir(parents=True)
        preflight = make_preflight(
            archive_path=args.archive,
            model_dir=args.model_dir,
            moment_source=args.moment_source,
            batch_size=args.batch_size,
        )
        _atomic_json(args.output_dir / "preflight.json", preflight)
        print(json.dumps(preflight, sort_keys=True))
    else:
        run_args = vars(args).copy()
        run_args.pop("preflight_only")
        run_args["archive_path"] = run_args.pop("archive")
        print(json.dumps(run_probe(**run_args), sort_keys=True))


if __name__ == "__main__":
    main()
