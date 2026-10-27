"""Measure nearest cross-split timestamp gaps on one preflight activity."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from scripts.audit_adl_participant_split import PROTOCOL_SHA, participant_split, protocol_digest
from scripts.audit_adl_sensor_identity import ARCHIVE_SHA


def nearest_cross_split_gaps(train, validation):
    train, validation = np.asarray(train, dtype=float), np.asarray(validation, dtype=float)
    if (train.ndim != 1 or validation.ndim != 1 or not len(train) or not len(validation)
            or not np.isfinite(train).all() or not np.isfinite(validation).all()):
        raise ValueError('both time arrays must be finite and nonempty')
    train = np.sort(train)
    positions = np.searchsorted(train, validation)
    left = train[np.clip(positions - 1, 0, len(train) - 1)]
    right = train[np.clip(positions, 0, len(train) - 1)]
    return np.minimum(np.abs(validation - left), np.abs(validation - right))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = Path('docs/pretrained_reuse/R2_ADL_PARTICIPANT_SPLIT_PROTOCOL_20261027.md')
    if protocol_digest(protocol) != PROTOCOL_SHA:
        raise ValueError('split protocol hash mismatch')
    with args.archive.open('rb') as f:
        if hashlib.file_digest(f, 'sha256').hexdigest() != ARCHIVE_SHA:
            raise ValueError('archive hash mismatch')
    import h5py

    with zipfile.ZipFile(args.archive) as archive:
        members = sorted(
            (item for item in archive.infolist() if item.filename.endswith('.hdf5')),
            key=lambda item: item.filename,
        )
        if len(members) != 5 or len({m.filename for m in members}) != 5:
            raise ValueError('expected five unique sensor files')
        member = members[0]
        if member.file_size > 128 * 2**20 or member.flag_bits & 1:
            raise ValueError('oversized or encrypted source')
        with h5py.File(io.BytesIO(archive.read(member)), 'r') as h:
            participants, trials, times = h['participant'][:], h['trial'][:], h['timestamp'][:]
    if len(participants) != len(times) or len(trials) != len(times):
        raise ValueError('unaligned metadata')
    if len(set(zip(map(int, participants), map(int, trials), strict=True))) != len(times):
        raise ValueError('duplicate participant/trial')
    groups = np.asarray([participant_split(value) for value in participants], dtype=object)
    train, validation = times[groups == 'train'], times[groups == 'validation']
    gaps = nearest_cross_split_gaps(train, validation)
    result = {
        'archive_sha256': ARCHIVE_SHA, 'split_protocol_sha256': PROTOCOL_SHA,
        'member': member.filename, 'train_events': len(train),
        'validation_events': len(validation),
        'unassigned_events': sum(group is None for group in groups),
        'validation_events_with_equal_train_timestamp': int((gaps == 0).sum()),
        'nearest_gap_quantiles_native_units': {
            str(quantile): float(np.quantile(gaps, quantile))
            for quantile in (0, 0.25, 0.5, 0.75, 1)
        },
        'limitations': 'per-row timestamps only; semantics, units, durations and clocks unverified',
        'training_or_validation_scoring': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
