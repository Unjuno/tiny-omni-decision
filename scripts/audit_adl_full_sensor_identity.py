"""Audit all pinned ADL sensor identities and participant splits, not waveforms."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from scripts.audit_adl_participant_split import PROTOCOL_SHA, participant_split
from scripts.download_adl_archives import PRIOR_ARCHIVES, hash_archive

RECORD_SHA = "c01bf60460981280c5063ea54c038d41c751bf3056220059c3896cbc5d2b97c6"
PLAN_SHA = "a387e837d9e14ea00fec20e72c07331fafd3f12e07661cf671da0f4216152dc2"
RECEIPT_SHA = "af284fe02e447a33194aa0fa871a0601dc0f067ec81137ceb9a17843379230aa"
MAX_MEMBER_BYTES = 600 * 2**20
DISK_RESERVE = 2 * 2**30
CHUNK_BYTES = 1024 * 1024


def sensor_index(member_name: str) -> int:
    name = Path(member_name).name
    try:
        token = name.rsplit("_multisensor-", 1)[1].removesuffix(".hdf5")
        index = int(token)
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid sensor member name: {member_name}") from exc
    if index not in range(1, 6):
        raise ValueError(f"sensor index must be 1..5: {member_name}")
    return index


def metadata_map(participant, trial, timestamp, sample_rate) -> dict:
    arrays = [np.asarray(value) for value in (participant, trial, timestamp, sample_rate)]
    if any(array.ndim != 1 for array in arrays) or len({array.shape for array in arrays}) != 1:
        raise ValueError("sensor metadata arrays must be aligned one-dimensional arrays")
    people, trials, times, rates = arrays
    if (
        not np.issubdtype(people.dtype, np.integer)
        or not np.issubdtype(trials.dtype, np.integer)
        or not np.isfinite(times).all()
        or not np.isfinite(rates).all()
        or np.any(rates <= 0)
    ):
        raise ValueError("invalid participant, trial, timestamp, or sample-rate metadata")
    rows = {}
    for person, event, time_value, rate in zip(people, trials, times, rates, strict=True):
        person_id, trial_id = int(person), int(event)
        if person_id < 0 or trial_id < 0:
            raise ValueError("participant and trial IDs must be non-negative")
        key = (person_id, trial_id)
        if key in rows:
            raise ValueError("duplicate participant/trial identity in one sensor")
        rows[key] = (float(time_value), float(rate))
    return rows


def compare_sensor_metadata(reference: dict, other: dict) -> dict:
    if reference.keys() != other.keys():
        raise ValueError("participant/trial identities differ across sensors")
    pairs = list(reference)
    return {
        "rows": len(pairs),
        "max_abs_timestamp_delta_native_units": max(
            (abs(reference[key][0] - other[key][0]) for key in pairs), default=0.0
        ),
        "max_abs_sample_rate_delta": max(
            (abs(reference[key][1] - other[key][1]) for key in pairs), default=0.0
        ),
    }


def extract_member_to_temp(archive: zipfile.ZipFile, member: zipfile.ZipInfo, temp_dir: Path):
    if member.flag_bits & 1 or member.file_size > MAX_MEMBER_BYTES:
        raise ValueError(f"encrypted or oversized HDF5 member: {member.filename}")
    path = temp_dir / "sensor.hdf5"
    digest = hashlib.sha256()
    copied = 0
    with archive.open(member, "r") as source, path.open("xb") as target:
        while chunk := source.read(CHUNK_BYTES):
            copied += len(chunk)
            if copied > member.file_size:
                raise ValueError(f"member expanded beyond declared size: {member.filename}")
            target.write(chunk)
            digest.update(chunk)
        target.flush()
    if copied != member.file_size:
        raise ValueError(f"member size mismatch: {member.filename}")
    return path, digest.hexdigest()


def verify_inputs(plan_path: Path, receipt_path: Path):
    plan_raw, receipt_raw = plan_path.read_bytes(), receipt_path.read_bytes()
    if hashlib.sha256(plan_raw).hexdigest() != PLAN_SHA:
        raise ValueError("download-plan SHA-256 mismatch")
    if hashlib.sha256(receipt_raw).hexdigest() != RECEIPT_SHA:
        raise ValueError("download-receipt SHA-256 mismatch")
    plan, receipt = json.loads(plan_raw), json.loads(receipt_raw)
    if plan.get("source_record_sha256") != RECORD_SHA:
        raise ValueError("source record hash mismatch")
    if receipt.get("summary", {}).get("status") != "ALL_ARCHIVES_MD5_VERIFIED":
        raise ValueError("download receipt is not complete")
    planned = plan.get("archives", [])
    if len(planned) != 25:
        raise ValueError("expected all 25 pinned activities")
    expected = []
    indices = set()
    for item in planned:
        key = item.get("key", "")
        prefix, separator, suffix = key.partition("_")
        checksum = item.get("md5", "")
        if (
            not separator
            or not prefix.isdecimal()
            or len(prefix) != 2
            or not suffix.endswith(".zip")
            or int(prefix) in indices
            or not isinstance(item.get("size"), int)
            or item["size"] <= 0
            or not checksum.startswith("md5:")
            or len(checksum) != 36
            or any(ch not in "0123456789abcdef" for ch in checksum[4:])
        ):
            raise ValueError(f"invalid pinned archive plan row: {key}")
        indices.add(int(prefix))
        expected.append({"key": key, "size": item["size"], "checksum": checksum})
    if indices != set(range(25)):
        raise ValueError("archive plan must cover activity numbers 00..24")
    expected.sort(key=lambda item: (item["size"], item["key"]))
    completed = {item.get("key") for item in receipt.get("downloads", [])}
    if len(completed) != 23 or completed != {row["key"] for row in expected} - set(PRIOR_ARCHIVES):
        raise ValueError("download receipt activity set mismatch")
    return expected, receipt


def receipt_archive_hashes(receipt: dict) -> dict[str, str]:
    hashes = {
        item.get("key"): item.get("sha256") for item in receipt.get("downloads", [])
    }
    hashes.update(
        {
            key: item.get("sha256")
            for key, item in receipt.get("prior_verified_archives", {}).items()
        }
    )
    if set(hashes) != {row["key"] for row in receipt.get("downloads", [])} | set(PRIOR_ARCHIVES):
        raise ValueError("receipt archive SHA-256 set mismatch")
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        for value in hashes.values()
    ):
        raise ValueError("receipt contains an invalid archive SHA-256")
    return hashes


def audit_activity(path: Path, row: dict, temp_parent: Path) -> dict:
    archive_hash = hash_archive(path, row["size"], row["checksum"])
    import h5py

    with zipfile.ZipFile(path) as archive:
        members = [
            member
            for member in archive.infolist()
            if member.filename.lower().endswith(".hdf5")
        ]
        by_sensor = {}
        for member in members:
            index = sensor_index(member.filename)
            if index in by_sensor:
                raise ValueError(f"duplicate sensor {index} in {path.name}")
            by_sensor[index] = member
        if set(by_sensor) != set(range(1, 6)):
            raise ValueError(f"expected exactly sensors 1..5 in {path.name}")

        reference = None
        sensors = []
        for index in range(1, 6):
            member = by_sensor[index]
            if shutil.disk_usage(temp_parent).free < member.file_size + DISK_RESERVE:
                raise OSError(f"insufficient disk reserve for {path.name} sensor {index}")
            with tempfile.TemporaryDirectory(prefix="adl-h5-", dir=temp_parent) as scratch:
                member_path, member_hash = extract_member_to_temp(
                    archive, member, Path(scratch)
                )
                with h5py.File(member_path, "r") as handle:
                    arrays = tuple(
                        handle[key][:]
                        for key in ("participant", "trial", "timestamp", "vibration_sample_rate")
                    )
                current = metadata_map(*arrays)
            if reference is None:
                reference = current
                alignment = {
                    "rows": len(current),
                    "max_abs_timestamp_delta_native_units": 0.0,
                    "max_abs_sample_rate_delta": 0.0,
                }
            else:
                alignment = compare_sensor_metadata(reference, current)
            sensors.append(
                {
                    "sensor_index": index,
                    "member": member.filename,
                    "member_size": member.file_size,
                    "member_sha256": member_hash,
                    **alignment,
                }
            )

    participants = [person for person, _ in reference]
    split_counts = {"train": 0, "validation": 0, "unassigned": 0}
    split_people = {"train": set(), "validation": set()}
    for person in participants:
        split = participant_split(person)
        if split is None:
            split_counts["unassigned"] += 1
        else:
            split_counts[split] += 1
            split_people[split].add(person)
    return {
        "activity": path.name,
        "archive_size": row["size"],
        "archive_publisher_md5": archive_hash["publisher_md5"],
        "archive_sha256": archive_hash["sha256"],
        "sensor_count": len(sensors),
        "event_rows": len(reference),
        "distinct_participants": len(set(participants)),
        "train_participants": len(split_people["train"]),
        "validation_participants": len(split_people["validation"]),
        "split_event_rows": split_counts,
        "coverage_pass": bool(split_people["train"] and split_people["validation"]),
        "sensors": sensors,
        "waveform_arrays_read": False,
    }


def run_audit(archives_dir: Path, plan_path: Path, receipt_path: Path, output_path: Path):
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")
    expected, receipt = verify_inputs(plan_path, receipt_path)
    receipt_hashes = receipt_archive_hashes(receipt)
    if hashlib.sha256(
        Path("docs/pretrained_reuse/R2_ADL_PARTICIPANT_SPLIT_PROTOCOL_20261027.md").read_bytes()
        .replace(b"\r\n", b"\n")
    ).hexdigest() != PROTOCOL_SHA:
        raise ValueError("participant split protocol hash mismatch")
    activities = []
    people_by_split = {"train": set(), "validation": set()}
    temp_parent = output_path.parent
    for row in expected:
        path = PRIOR_ARCHIVES.get(row["key"], archives_dir / row["key"])
        result = audit_activity(path, row, temp_parent)
        if result["archive_sha256"] != receipt_hashes[row["key"]]:
            raise ValueError(f"archive SHA-256 differs from pinned receipt: {row['key']}")
        activities.append(result)
        # Re-read only aggregate split memberships; IDs are never serialized.
        with zipfile.ZipFile(path) as archive:
            first_sensor = min(
                (item for item in archive.infolist() if item.filename.lower().endswith(".hdf5")),
                key=lambda item: sensor_index(item.filename),
            )
            import h5py

            with tempfile.TemporaryDirectory(prefix="adl-split-", dir=temp_parent) as scratch:
                member_path, _ = extract_member_to_temp(archive, first_sensor, Path(scratch))
                with h5py.File(member_path, "r") as handle:
                    people = handle["participant"][:]
            for person in set(map(int, people)):
                split = participant_split(person)
                if split is not None:
                    people_by_split[split].add(person)
    overlap = people_by_split["train"] & people_by_split["validation"]
    if overlap:
        raise ValueError("participant appears in both fixed splits")
    result = {
        "source_record_sha256": RECORD_SHA,
        "download_plan_sha256": PLAN_SHA,
        "download_receipt_sha256": RECEIPT_SHA,
        "participant_split_protocol_sha256": PROTOCOL_SHA,
        "activity_count": len(activities),
        "sensor_member_count": sum(item["sensor_count"] for item in activities),
        "activities_with_both_splits": sum(item["coverage_pass"] for item in activities),
        "unassigned_rows": sum(item["split_event_rows"]["unassigned"] for item in activities),
        "distinct_train_participants": len(people_by_split["train"]),
        "distinct_validation_participants": len(people_by_split["validation"]),
        "participant_overlap": 0,
        "activities": activities,
        "scope": (
            "pinned source metadata and cross-sensor identity only; "
            "no model or waveform evaluation"
        ),
        "physical_clock_synchronization": "not established by shared row identity/timestamps",
        "waveform_arrays_read": False,
        "sealed_audit_loaded": False,
        "receipt_summary": receipt["summary"],
    }
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(args.archives_dir, args.plan, args.receipt, args.output)
    print(
        json.dumps(
            {key: result[key] for key in (
                "activity_count", "sensor_member_count", "activities_with_both_splits",
                "unassigned_rows", "distinct_train_participants",
                "distinct_validation_participants",
                "participant_overlap", "waveform_arrays_read",
            )},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
