"""Compare cross-activity train/validation timestamp metadata in two pinned archives."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from scripts.audit_adl_cross_split_time import nearest_cross_split_gaps
from scripts.audit_adl_participant_split import PROTOCOL_SHA, participant_split, protocol_digest

ARCHIVES = {
    "10_light-off.zip": "3664af53b9b4fccc9abf06d9922f66d24ee81967022d9036bd355a01462ec8e0",
    "23_take-from-stove.zip": "04bf32f55c93f54c49525d2b0581cb51f14c1ec302afb56b70a995fa4aa8f706",
}


def compare(train, validation):
    gaps = nearest_cross_split_gaps(train, validation)
    return {
        "train_events": len(train),
        "validation_events": len(validation),
        "equal_timestamp_validation_events": int((gaps == 0).sum()),
        "nearest_gap_quantiles_native_units": {
            str(q): float(np.quantile(gaps, q)) for q in (0, 0.25, 0.5, 0.75, 1)
        },
    }


def ordered_hdf5_members(infos):
    return sorted(
        (info for info in infos if info.filename.endswith(".hdf5")), key=lambda info: info.filename
    )


def read_metadata(path, expected_hash):
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected_hash:
            raise ValueError("archive hash mismatch")
    import h5py

    with zipfile.ZipFile(path) as archive:
        members = ordered_hdf5_members(archive.infolist())
        if len(members) != 5 or len({i.filename for i in members}) != 5:
            raise ValueError("expected five unique HDF5 members")
        first = members[0]
        if first.file_size > 128 * 2**20 or first.flag_bits & 1:
            raise ValueError("oversized or encrypted member")
        with h5py.File(io.BytesIO(archive.read(first)), "r") as h:
            people, trials, times = h["participant"][:], h["trial"][:], h["timestamp"][:]
    if (
        people.ndim != 1
        or people.shape != trials.shape
        or people.shape != times.shape
        or not np.isfinite(times).all()
    ):
        raise ValueError("invalid metadata")
    if len(set(zip(map(int, people), map(int, trials), strict=True))) != len(people):
        raise ValueError("duplicate participant/trial")
    groups = np.asarray([participant_split(p) for p in people], dtype=object)
    return {
        "train": times[groups == "train"],
        "validation": times[groups == "validation"],
        "unassigned": int(sum(g is None for g in groups)),
        "member": first.filename,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output exists")
    protocol = Path("docs/pretrained_reuse/R2_ADL_PARTICIPANT_SPLIT_PROTOCOL_20261027.md")
    if protocol_digest(protocol) != PROTOCOL_SHA:
        raise ValueError("split protocol hash mismatch")
    if {p.name for p in args.archives} != set(ARCHIVES):
        raise ValueError("exactly the two pinned activities are required")
    data = {p.name: read_metadata(p, ARCHIVES[p.name]) for p in args.archives}
    first, second = sorted(data)
    result = {
        "archives_sha256": ARCHIVES,
        "split_protocol_sha256": PROTOCOL_SHA,
        "protocol_sha256_lf": protocol_digest(
            Path("docs/pretrained_reuse/R2_ADL_CROSS_ACTIVITY_TIME_PROTOCOL_20261027.md")
        ),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "members": {k: v["member"] for k, v in data.items()},
        "unassigned_events": {k: v["unassigned"] for k, v in data.items()},
        "directions": {
            f"{first}:train->{second}:validation": compare(
                data[first]["train"], data[second]["validation"]
            ),
            f"{second}:train->{first}:validation": compare(
                data[second]["train"], data[first]["validation"]
            ),
        },
        "waveforms_read": False,
        "model_run": False,
        "sealed_audit_opened": False,
        "limitations": "native timestamps only; no event-duration or session disjointness proof",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
