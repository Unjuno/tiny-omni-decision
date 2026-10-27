"""Audit audio/sensor group identities without reading the waveforms."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

ARCHIVE_SHA = '04bf32f55c93f54c49525d2b0581cb51f14c1ec302afb56b70a995fa4aa8f706'


def _as_mapping(rows):
    participants, trials, times = (np.asarray(v) for v in rows)
    if (participants.ndim != 1 or participants.shape != trials.shape
            or participants.shape != times.shape or not np.isfinite(times).all()):
        raise ValueError('unaligned or nonfinite metadata')
    if (not np.issubdtype(participants.dtype, np.integer)
            or not np.issubdtype(trials.dtype, np.integer)):
        raise ValueError('participant and trial IDs must be integers')
    pairs = list(zip(map(int, participants), map(int, trials), strict=True))
    if len(set(pairs)) != len(pairs):
        raise ValueError('duplicate participant/trial identity')
    return dict(zip(pairs, map(float, times), strict=True))


def compare_sensor_rows(reference, other):
    ref, sensor = _as_mapping(reference), _as_mapping(other)
    common = ref.keys() & sensor.keys()
    delta = [abs(sensor[key] - ref[key]) for key in common]
    return {
        'common_pairs': len(common),
        'reference_only_pairs': len(ref.keys() - sensor.keys()),
        'sensor_only_pairs': len(sensor.keys() - ref.keys()),
        'max_abs_timestamp_delta_native_units': max(delta) if delta else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.archive.open('rb') as f:
        if hashlib.file_digest(f, 'sha256').hexdigest() != ARCHIVE_SHA:
            raise ValueError('archive SHA-256 mismatch')
    import h5py

    records = []
    reference = None
    with zipfile.ZipFile(args.archive) as archive:
        members = sorted(
            (m for m in archive.infolist() if m.filename.lower().endswith('.hdf5')),
            key=lambda m: m.filename,
        )
        if len(members) != 5 or len({m.filename for m in members}) != 5:
            raise ValueError('expected five distinct HDF5 members')
        for member in members:
            if member.file_size > 128 * 2**20 or member.flag_bits & 1:
                raise ValueError('member is encrypted or exceeds size cap')
            data = archive.read(member)
            if len(data) != member.file_size:
                raise ValueError('HDF5 member size mismatch')
            with h5py.File(io.BytesIO(data), 'r') as h:
                rows = (h['participant'][:], h['trial'][:], h['timestamp'][:])
                rates = h['vibration_sample_rate'][:]
                if (rates.shape != rows[0].shape or not np.isfinite(rates).all()
                        or np.any(rates <= 0)):
                    raise ValueError('invalid vibration rates')
                if reference is None:
                    reference = rows
                comparison = compare_sensor_rows(reference, rows)
                records.append({
                    'member': member.filename,
                    'member_sha256': hashlib.sha256(data).hexdigest(),
                    'rows': len(rows[0]), 'participants': len(set(map(int, rows[0]))),
                    'vibration_rate_min': float(rates.min()),
                    'vibration_rate_max': float(rates.max()), **comparison,
                })
    result = {
        'archive_sha256': ARCHIVE_SHA, 'records': records,
        'scope': 'one activity; participant/trial/timestamp/rate metadata only',
        'timestamp_units_and_physical_synchronization': 'not verified',
        'training_or_validation': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
