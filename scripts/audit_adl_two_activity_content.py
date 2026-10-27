"""Check exact audio and vibration content identity across a fixed participant split."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.audit_adl_cross_activity_time import ARCHIVES, ordered_hdf5_members
from scripts.audit_adl_participant_split import PROTOCOL_SHA, participant_split, protocol_digest

CHANNELS = ("audio", "vibration-x", "vibration-y", "vibration-z")


def content_digest(value):
    samples = np.asarray(value)
    if samples.ndim != 1 or samples.dtype.kind != "i" or samples.dtype.itemsize != 2:
        raise ValueError("expected one-dimensional signed int16 data")
    samples = samples.astype("<i2", copy=False)
    digest = hashlib.sha256(len(samples).to_bytes(8, "big") + samples.tobytes()).hexdigest()
    return digest, bool(len(samples) and np.all(samples == samples[0])), len(samples) == 0


def overlap_summary(train, validation):
    train_counts, val_counts = Counter(train), Counter(validation)
    common = train_counts.keys() & val_counts.keys()
    return {
        "unique_train_hashes": len(train_counts),
        "unique_validation_hashes": len(val_counts),
        "common_hashes": len(common),
        "train_rows_with_validation_match": sum(train_counts[key] for key in common),
        "validation_rows_with_train_match": sum(val_counts[key] for key in common),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output exists")
    if {path.name for path in args.archives} != set(ARCHIVES):
        raise ValueError("exactly the two pinned activities are required")
    if (
        protocol_digest("docs/pretrained_reuse/R2_ADL_PARTICIPANT_SPLIT_PROTOCOL_20261027.md")
        != PROTOCOL_SHA
    ):
        raise ValueError("split protocol hash mismatch")
    import h5py

    hashes = {channel: {"train": [], "validation": []} for channel in CHANNELS}
    stats = {channel: {"constant": 0, "empty": 0} for channel in CHANNELS}
    row_counts = {"train": 0, "validation": 0}
    for path in args.archives:
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != ARCHIVES[path.name]:
                raise ValueError("archive hash mismatch")
        with zipfile.ZipFile(path) as archive:
            members = ordered_hdf5_members(archive.infolist())
            if len(members) != 5 or len({item.filename for item in members}) != 5:
                raise ValueError("expected five unique members")
            member = members[0]
            if member.file_size > 128 * 2**20 or member.flag_bits & 1:
                raise ValueError("oversized or encrypted member")
            payload = archive.read(member)
        with h5py.File(io.BytesIO(payload), "r") as source:
            people, trials = source["participant"][:], source["trial"][:]
            if len(people) != len(trials) or any(len(source[c]) != len(people) for c in CHANNELS):
                raise ValueError("misaligned rows")
            if len(set(zip(map(int, people), map(int, trials), strict=True))) != len(people):
                raise ValueError("duplicate participant/trial")
            for row, person in enumerate(people):
                split = participant_split(person)
                if split is None:
                    raise ValueError("unassigned participant cannot be compared")
                row_counts[split] += 1
                for channel in CHANNELS:
                    digest, constant, empty = content_digest(source[channel][row])
                    hashes[channel][split].append(digest)
                    stats[channel]["constant"] += int(constant)
                    stats[channel]["empty"] += int(empty)
    results = {
        channel: {**overlap_summary(parts["train"], parts["validation"]), **stats[channel]}
        for channel, parts in hashes.items()
    }
    result = {
        "archives_sha256": ARCHIVES,
        "split_protocol_sha256": PROTOCOL_SHA,
        "content_protocol_sha256_lf": protocol_digest(
            "docs/pretrained_reuse/R2_ADL_TWO_ACTIVITY_CONTENT_PROTOCOL_20261027.md"
        ),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "row_counts": row_counts,
        "channels": results,
        "raw_waveforms_written": False,
        "training_or_scoring": False,
        "sealed_audit_opened": False,
        "limitations": "exact channel identity only; not session or partial-clip overlap",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
