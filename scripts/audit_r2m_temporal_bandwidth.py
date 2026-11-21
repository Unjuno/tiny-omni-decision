"""Inspect the fixed 128-token vibration compression on TRAIN rows only."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from tiny_omni_decision.r2m_temporal_bandwidth import (
    effective_uniform_sample_rate,
    out_of_band_energy_fraction,
)

SPLIT_SHA256 = "0f1539f3ca4fa59fb854f62def1c538e1f5851576444b678af4aa050a393a6f0"
PAIR_INDEX_SHA256 = "d1e1c74a1114fc0d071f71a50876b852e8d3fce8919fde9696368d4ebf365276"
METADATA_SHA256 = "faee790bfaa53016e7c92f0ac545f0a7dfe23cd126fa7e1287b10d8d8b0282c0"
DOWNLOAD_INDEX_SHA256 = "715dd57fdf1967767f28f2b2246c57232dfba3e74ea548d46c83e68b882402d4"
REPO = Path(__file__).resolve().parents[1]
SPLIT_PATH = REPO / "manifests/pretrained_reuse/r2m-audio-vibration-split-v2-20261120.json"
SEED = 17
TARGET_TOKENS = 128
COUNTS_TO_G = 2.5 / 32768.0


def stable_sample_id(row: dict[str, object]) -> str:
    return (
        f"{row['activity']}|participant={int(row['participant_id'])}|trial={int(row['trial_id'])}"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: dict[str, object]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def select_train_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Select one deterministic training trial per activity/participant."""
    buckets: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    seen_ids: set[str] = set()
    for row in rows:
        identifier = stable_sample_id(row)
        if identifier in seen_ids:
            raise ValueError("metadata must have unique nonempty sample IDs")
        seen_ids.add(identifier)
        if row.get("split") != "train":
            continue
        participant = row.get("participant_id")
        if type(participant) is not int or participant not in {2, 4, 7, 8, 10, 13, 14}:
            raise ValueError("TRAIN metadata contains a participant outside split v2")
        buckets[(str(row["activity"]), participant)].append(row)

    selected = []
    for key in sorted(buckets):
        bucket = buckets[key]
        selected.append(
            min(
                bucket,
                key=lambda row: hashlib.sha256(
                    f"r2m-temporal-bandwidth-v1|seed={SEED}|{stable_sample_id(row)}".encode()
                ).hexdigest(),
            )
        )
    if not selected:
        raise ValueError("no TRAIN rows selected")
    return selected


def verify_pair_index(pair_path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    split_payload = {key: value for key, value in split.items() if key != "manifest_sha256"}
    if (
        split.get("manifest_sha256") != SPLIT_SHA256
        or canonical_hash(split_payload) != SPLIT_SHA256
    ):
        raise ValueError("participant split differs from the frozen R2-M split")
    if set(split.get("splits", {}).get("train", [])) != {2, 4, 7, 8, 10, 13, 14}:
        raise ValueError("TRAIN participant list differs from frozen split v2")
    artifact = json.loads(pair_path.read_text(encoding="utf-8"))
    payload = {key: value for key, value in artifact.items() if key != "manifest_sha256"}
    if canonical_hash(payload) != PAIR_INDEX_SHA256:
        raise ValueError("pair index differs from the frozen R2-M artifact")
    if artifact.get("sealed_audit_accessed") is not False:
        raise ValueError("pair index does not establish sealed-audit isolation")
    metadata_path = pair_path.parent / "allowed-trial-metadata.json"
    if sha256_file(metadata_path) != METADATA_SHA256:
        raise ValueError("allowed trial metadata differs from the frozen pair index")
    rows = json.loads(metadata_path.read_text(encoding="utf-8"))
    return artifact, rows


def audit_selected_rows(
    selected: list[dict[str, object]],
    source_root: Path,
    receipt_path: Path,
) -> list[dict[str, object]]:
    import h5py

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("download_index_sha256") != DOWNLOAD_INDEX_SHA256:
        raise ValueError("download receipt differs from the frozen source")
    expected_files = {item["file"]: item for item in receipt["files"]}
    by_activity: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in selected:
        if row.get("split") != "train":
            raise ValueError("selected row is outside TRAIN")
        by_activity[str(row["activity"])].append(row)

    outputs: list[dict[str, object]] = []
    for activity, rows in sorted(by_activity.items()):
        archive_name = f"{activity}.zip"
        expected = expected_files.get(archive_name)
        if not expected or expected.get("verification") != "PASS":
            raise ValueError(f"archive is not in the verified receipt: {archive_name}")
        archive_path = source_root / archive_name
        if (
            archive_path.stat().st_size != expected["bytes"]
            or sha256_file(archive_path) != expected["sha256"]
        ):
            raise ValueError(f"archive hash differs from the receipt: {archive_name}")
        lookup = {(int(row["participant_id"]), int(row["trial_id"])): row for row in rows}
        with zipfile.ZipFile(archive_path) as archive:
            members = [name for name in archive.namelist() if name.endswith("_multisensor-1.hdf5")]
            if len(members) != 1:
                raise ValueError(f"expected one multisensor-1 member: {archive_name}")
            raw = archive.read(members[0])
        with h5py.File(io.BytesIO(raw), "r") as data:
            participants = data["participant"][:]
            trials = data["trial"][:]
            found: set[tuple[int, int]] = set()
            for index, (pid_value, trial_value) in enumerate(
                zip(participants, trials, strict=True)
            ):
                key = (int(pid_value), int(trial_value))
                row = lookup.get(key)
                if row is None:
                    # In particular, no sealed/quarantined signal arrays are read.
                    continue
                if row.get("split") != "train":
                    raise ValueError("non-TRAIN signal access was selected")
                found.add(key)
                rate = float(data["vibration_sample_rate"][index])
                axes = [
                    np.asarray(data[f"vibration-{axis}"][index], dtype=np.float64) for axis in "xyz"
                ]
                if (
                    not math.isfinite(rate)
                    or rate <= 0
                    or any(axis.ndim != 1 or axis.size < 2 for axis in axes)
                    or len({axis.size for axis in axes}) != 1
                ):
                    raise ValueError(f"invalid TRAIN vibration row: {row['sample_id']}")
                audio_samples = int(row["audio_samples"])
                duration = audio_samples / float(row["audio_sample_rate_hz"])
                target_rate = effective_uniform_sample_rate(TARGET_TOKENS, duration)
                means = [float(axis.mean()) for axis in axes]
                stds = [float(axis.std()) for axis in axes]
                fractions = [out_of_band_energy_fraction(axis, rate, target_rate) for axis in axes]
                gravity = [value * COUNTS_TO_G for value in means]
                outputs.append(
                    {
                        "sample_id": stable_sample_id(row),
                        "split": "train",
                        "activity": activity,
                        "participant_id": key[0],
                        "trial_id": key[1],
                        "duration_seconds": duration,
                        "raw_vibration_rate_hz": rate,
                        "raw_vibration_samples": int(axes[0].size),
                        "fixed_feature_tokens": TARGET_TOKENS,
                        "effective_feature_rate_hz": target_rate,
                        "effective_feature_nyquist_hz": target_rate / 2,
                        "raw_axis_mean_counts": means,
                        "raw_axis_std_counts": stds,
                        "axis_mean_g_using_published_range": gravity,
                        "raw_dynamic_energy_above_feature_nyquist_fraction": fractions,
                    }
                )
        if found != set(lookup):
            raise ValueError(f"selected TRAIN rows missing from {archive_name}")
    outputs.sort(key=lambda row: str(row["sample_id"]))
    if len(outputs) != len(selected):
        raise ValueError("audited sample count differs from deterministic selection")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--pair-index", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit("refusing to overwrite a nonempty output directory")
    pair_index, rows = verify_pair_index(args.pair_index)
    selected = select_train_rows(rows)
    audit = audit_selected_rows(selected, args.source_root, args.receipt)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "selected-train-samples.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in audit),
        encoding="utf-8",
        newline="\n",
    )
    durations = np.asarray([row["duration_seconds"] for row in audit])
    nyquist = np.asarray([row["effective_feature_nyquist_hz"] for row in audit])
    energy = np.asarray(
        [
            value
            for row in audit
            for value in row["raw_dynamic_energy_above_feature_nyquist_fraction"]
        ]
    )
    summary = {
        "schema": "r2m_temporal_bandwidth_audit_v1",
        "source": "10.5281/zenodo.7937591",
        "split_sha256": SPLIT_SHA256,
        "pair_index_sha256": pair_index["manifest_sha256"],
        "metadata_sha256": METADATA_SHA256,
        "seed": SEED,
        "selection": "one SHA-256-ranked TRAIN trial per available activity x TRAIN participant",
        "selected_sample_count": len(audit),
        "sealed_audit_loaded": False,
        "quarantined_participant_signal_loaded": False,
        "validation_signal_loaded": False,
        "vibration_feature_tokens": TARGET_TOKENS,
        "duration_seconds": {
            "min": float(durations.min()),
            "median": float(np.median(durations)),
            "max": float(durations.max()),
        },
        "effective_feature_nyquist_hz": {
            "min": float(nyquist.min()),
            "median": float(np.median(nyquist)),
            "max": float(nyquist.max()),
        },
        "raw_dynamic_energy_above_effective_nyquist_fraction": {
            "axis_sample_median": float(np.median(energy)),
            "axis_sample_p10": float(np.quantile(energy, 0.1)),
            "axis_sample_p90": float(np.quantile(energy, 0.9)),
        },
    }
    # Write the row artifact first, then pin its hash in the summary.
    jsonl_path = args.output_dir / "selected-train-samples.jsonl"
    summary["selected_rows_sha256"] = sha256_file(jsonl_path)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
