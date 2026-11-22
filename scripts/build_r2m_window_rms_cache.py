"""Build a separate RMS-envelope cache without rerunning the frozen audio encoder."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np

from scripts.audit_r2m_temporal_bandwidth import (
    DOWNLOAD_INDEX_SHA256,
    METADATA_SHA256,
    PAIR_INDEX_SHA256,
    SPLIT_PATH,
    SPLIT_SHA256,
    canonical_hash,
    sha256_file,
    stable_sample_id,
)
from tiny_omni_decision.r2m_temporal_bandwidth import windowed_rms_tokens

SOURCE_FEATURE_INDEX_SHA256 = "61758bebaf53ad810737d31e7955820bf7094e2166460c5ce595ea0c475c6498"
TOKENIZATION = "per-trial-mean-centered-window-rms-128-v1"
ALLOWED_SPLITS = {"train", "calibration", "validation"}


def load_split_participants() -> dict[int, str]:
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    payload = {key: value for key, value in split.items() if key != "manifest_sha256"}
    if split.get("manifest_sha256") != SPLIT_SHA256 or canonical_hash(payload) != SPLIT_SHA256:
        raise ValueError("participant split differs from the frozen R2-M split")
    mapping: dict[int, str] = {}
    for split_name in ("train", "calibration", "validation", "sealed_audit"):
        for participant in split["splits"][split_name]:
            if type(participant) is not int or participant in mapping:
                raise ValueError("invalid or duplicate participant in split v2")
            mapping[participant] = split_name
    if set(mapping) != {1, 2, 4, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17}:
        raise ValueError("participant inventory differs from frozen split v2")
    return mapping


def tensor_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def build_cache(
    source_cache: Path,
    source_root: Path,
    pair_index_dir: Path,
    receipt_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError("building the R2-M sensor cache requires h5py") from error
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite cache output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    output_features = output_dir / "features"
    output_features.mkdir()

    pair_path = pair_index_dir / "pair-index.json"
    metadata_path = pair_index_dir / "allowed-trial-metadata.json"
    if sha256_file(pair_path) != "33664b32205626fe1ec2102961e82163d2f55586f6c0567c9f1c80c0f4d3e7bf":
        raise ValueError("pair-index file differs from the frozen R2-M artifact")
    if sha256_file(metadata_path) != METADATA_SHA256:
        raise ValueError("allowed-trial metadata differs from the frozen R2-M artifact")
    pair = json.loads(pair_path.read_text(encoding="utf-8"))
    pair_payload = {key: value for key, value in pair.items() if key != "manifest_sha256"}
    if (
        canonical_hash(pair_payload) != PAIR_INDEX_SHA256
        or pair.get("sealed_audit_accessed") is not False
    ):
        raise ValueError("pair-index integrity or sealed-audit gate failed")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_by_id = {stable_sample_id(row): row for row in metadata}
    if len(metadata_by_id) != len(metadata):
        raise ValueError("duplicate sample identity in frozen allowed metadata")

    participant_to_split = load_split_participants()
    source_index_path = source_cache / "feature-cache-index.json"
    if sha256_file(source_index_path) != SOURCE_FEATURE_INDEX_SHA256:
        raise ValueError("source feature-cache index differs from the frozen cache")
    source_index = json.loads(source_index_path.read_text(encoding="utf-8"))
    if (
        source_index.get("pair_index_sha256") != PAIR_INDEX_SHA256
        or source_index.get("sealed_audit_accessed") is not False
        or len(source_index.get("archives", [])) != 24
    ):
        raise ValueError("source feature cache fails frozen pair/scope checks")

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("download_index_sha256") != DOWNLOAD_INDEX_SHA256:
        raise ValueError("source archive receipt differs from the frozen release")
    archive_receipts = {row["file"]: row for row in receipt["files"]}
    split_counts = {name: 0 for name in ALLOWED_SPLITS}
    sidecars: list[dict[str, object]] = []
    total_samples = 0
    for source_sidecar_path in sorted((source_cache / "features").glob("*.json")):
        source_sidecar = json.loads(source_sidecar_path.read_text(encoding="utf-8"))
        activity = source_sidecar_path.stem
        if source_sidecar != {
            row["activity_archive"][:-4]: row for row in source_index["archives"]
        }.get(activity):
            raise ValueError(f"source sidecar differs from frozen index: {activity}")
        source_npz = source_sidecar_path.with_suffix(".npz")
        if sha256_file(source_npz) != source_sidecar.get("feature_sha256"):
            raise ValueError(f"source feature hash mismatch: {source_npz.name}")
        with np.load(source_npz, allow_pickle=False) as cached:
            ids = cached["sample_ids"].astype(str)
            splits = cached["splits"].astype(str)
            participants = cached["participant_ids"].astype(int)
            trials = cached["trial_ids"].astype(int)
            activities = cached["activities"]
            durations = cached["durations_seconds"]
            audio_offsets = cached["audio_offsets"]
            # NPZ compression is per array, so the old flattened audio feature
            # member is decompressed as a whole; only TRAIN spans are retained
            # or passed to the CV loader below. Old sensor features are never read.
            audio_values = cached["audio_values"]
            if len(ids) != len(set(ids.tolist())):
                raise ValueError(f"invalid row identities: {source_npz.name}")
            if audio_offsets.shape != (len(ids) + 1,) or audio_offsets[0] != 0:
                raise ValueError(f"invalid audio offsets: {source_npz.name}")
            if audio_offsets[-1] != len(audio_values) or np.any(np.diff(audio_offsets) <= 0):
                raise ValueError(f"invalid audio feature spans: {source_npz.name}")
        train_indices = [index for index, split in enumerate(splits) if split == "train"]
        lookup: dict[tuple[int, int], tuple[int, str]] = {}
        for index in train_indices:
            identifier = str(ids[index])
            metadata_row = metadata_by_id.get(str(identifier))
            participant = int(participants[index])
            declared_split = participant_to_split.get(participant)
            observed_split = str(splits[index])
            if (
                metadata_row is None
                or observed_split != "train"
                or declared_split != observed_split
                or metadata_row["split"] != observed_split
                or metadata_row["activity"] != activity
                or int(metadata_row["trial_id"]) != int(trials[index])
            ):
                raise ValueError(f"cache row violates frozen TRAIN/CAL/VAL identity: {identifier}")
            key = (participant, int(trials[index]))
            if key in lookup:
                raise ValueError(f"duplicate activity trial in source cache: {key}")
            lookup[key] = (index, observed_split)
        targets = {key: row_index for key, (row_index, _) in lookup.items()}

        archive_name = f"{activity}.zip"
        archive_receipt = archive_receipts.get(archive_name)
        if not archive_receipt or archive_receipt.get("verification") != "PASS":
            raise ValueError(f"source archive absent from verified receipt: {archive_name}")
        archive_path = source_root / archive_name
        if (
            archive_path.stat().st_size != archive_receipt["bytes"]
            or sha256_file(archive_path) != archive_receipt["sha256"]
        ):
            raise ValueError(f"source archive hash mismatch: {archive_name}")

        sensors_by_index: dict[int, np.ndarray] = {}
        found: set[tuple[int, int]] = set()
        with zipfile.ZipFile(archive_path) as archive:
            members = [name for name in archive.namelist() if name.endswith("_multisensor-1.hdf5")]
            if len(members) != 1:
                raise ValueError(f"expected one multisensor-1 member: {archive_name}")
            raw_hdf5 = archive.read(members[0])
        with h5py.File(io.BytesIO(raw_hdf5), "r") as data:
            source_participants = data["participant"][:]
            source_trials = data["trial"][:]
            for source_row, (pid_raw, trial_raw) in enumerate(
                zip(source_participants, source_trials, strict=True)
            ):
                key = (int(pid_raw), int(trial_raw))
                target_index = targets.get(key)
                if target_index is None:
                    # Do not dereference signals for sealed, quarantined, or
                    # otherwise unselected source records.
                    continue
                _, split = lookup[key]
                if split not in ALLOWED_SPLITS:
                    raise ValueError("non-iterative split reached the signal reader")
                axes = [
                    np.asarray(data[f"vibration-{axis}"][source_row], dtype=np.float32)
                    for axis in "xyz"
                ]
                rate = float(data["vibration_sample_rate"][source_row])
                if (
                    not np.isfinite(rate)
                    or rate <= 0
                    or any(axis.ndim != 1 or axis.size < 128 for axis in axes)
                    or len({axis.size for axis in axes}) != 1
                ):
                    raise ValueError(f"invalid vibration row for {activity}|{key}")
                sensors_by_index[target_index] = windowed_rms_tokens(np.stack(axes, axis=1), 128)
                found.add(key)
        if found != set(lookup):
            raise ValueError(f"source/cache ID mismatch for {activity}: {len(found)} rows")

        if set(sensors_by_index) != set(train_indices):
            raise ValueError(f"not every TRAIN sensor row was rebuilt: {activity}")
        train_audio = [
            audio_values[audio_offsets[index] : audio_offsets[index + 1]]
            for index in train_indices
        ]
        flattened_audio = np.concatenate(train_audio, axis=0).astype(np.float32, copy=False)
        original_train_audio_sha = tensor_sha256(flattened_audio)
        new_offsets = np.zeros(len(train_indices) + 1, dtype=np.int64)
        new_offsets[1:] = np.cumsum([len(item) for item in train_audio])
        train_ids = ids[train_indices]
        train_activities = activities[train_indices].astype(str)
        train_participants = participants[train_indices]
        train_trials = trials[train_indices]
        train_durations = durations[train_indices]
        arrays = {
            "sample_ids": train_ids,
            "participant_ids": train_participants.astype(np.int16, copy=False),
            "trial_ids": train_trials.astype(np.int32, copy=False),
            "splits": np.asarray(["train"] * len(train_indices), dtype="U16"),
            "activities": np.asarray(train_activities, dtype="U64"),
            "durations_seconds": train_durations.astype(np.float32, copy=False),
            "audio_values": flattened_audio,
            "audio_offsets": new_offsets,
            "sensor_values": np.stack([sensors_by_index[index] for index in train_indices]).astype(
                np.float32, copy=False
            ),
        }
        if tensor_sha256(arrays["audio_values"]) != original_train_audio_sha:
            raise AssertionError("TRAIN audio features changed while rebuilding sensor cache")
        output_npz = output_features / source_npz.name
        temporary_npz = output_npz.with_suffix(".partial.npz")
        np.savez_compressed(temporary_npz, **arrays)
        temporary_npz.replace(output_npz)
        with np.load(output_npz, allow_pickle=False) as written:
            if tensor_sha256(written["audio_values"]) != original_train_audio_sha:
                raise AssertionError("serialized TRAIN audio features changed")

        sidecar = dict(source_sidecar)
        sidecar.update(
            {
                "samples": len(train_indices),
                "audio_feature_shape": [int(flattened_audio.shape[0]), 384],
                "sensor_feature_shape": [len(train_indices), 128, 3],
                "feature_split": "train-only",
                "sensor_tokenization": TOKENIZATION,
                "source_feature_sha256": source_sidecar["feature_sha256"],
                "feature_sha256": sha256_file(output_npz),
                "audio_feature_values_sha256": original_train_audio_sha,
            }
        )
        sidecar_path = output_npz.with_suffix(".json")
        sidecar_path.write_text(
            json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        sidecars.append(sidecar)
        split_counts["train"] += len(train_indices)
        total_samples += len(train_indices)
        print(f"rebuilt {activity}: {len(train_indices)} TRAIN rows", flush=True)

    if total_samples != int(source_index["sample_count_by_split"]["train"]):
        raise ValueError("rebuilt row count differs from frozen TRAIN feature cache")
    source_code_hash = sha256_file(Path(__file__).resolve())
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    summary = {
        "schema": "r2m_frozen_feature_cache_v1",
        "sensor_tokenization": TOKENIZATION,
        "source_feature_cache_index_sha256": SOURCE_FEATURE_INDEX_SHA256,
        "pair_index_sha256": PAIR_INDEX_SHA256,
        "source_commit": source_commit,
        "builder_sha256": source_code_hash,
        "encoder_parameters": source_index["encoder_parameters"],
        "encoder_trainable_parameters": 0,
        "sample_count_by_split": split_counts,
        "archives": sidecars,
        "sealed_audit_accessed": False,
        "feature_signal_splits_loaded": ["train"],
    }
    index_path = output_dir / "feature-cache-index.json"
    index_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "status": "complete",
        "feature_cache_index_sha256": sha256_file(index_path),
        "sample_count_by_split": split_counts,
        "feature_samples": total_samples,
        "sensor_tokenization": TOKENIZATION,
        "sealed_audit_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--pair-index-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_cache(
        args.source_cache,
        args.source_root,
        args.pair_index_dir,
        args.receipt,
        args.output_dir,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
