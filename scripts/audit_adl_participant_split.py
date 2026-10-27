"""Check a fixed participant split against downloaded activity metadata."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import numbers
import zipfile
from pathlib import Path

import numpy as np

from scripts.audit_adl_sensor_identity import compare_sensor_rows

PROTOCOL_SHA = '9c301d4a3acc37711d25943e7165533410bb3297630eef19c7f6431d2daaf5f8'


def protocol_digest(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def participant_split(participant):
    if isinstance(participant, bool) or not isinstance(participant, numbers.Integral):
        raise ValueError('participant ID must be an integer')
    participant = int(participant)
    if participant < 0:
        raise ValueError('negative participant ID')
    if participant == 999:
        return None
    digest = hashlib.sha256(f'adl-participant-v1:17:{participant}'.encode()).digest()
    return 'train' if int.from_bytes(digest[:8], 'big') % 100 < 70 else 'validation'


def summarize_activity(participants, trials):
    if len(participants) != len(trials):
        raise ValueError('participants and trials are not aligned')
    split_people = {'train': set(), 'validation': set()}
    event_counts = {'train': 0, 'validation': 0}
    seen, unassigned = set(), 0
    for person, trial in zip(participants, trials, strict=True):
        split = participant_split(person)
        if isinstance(trial, bool) or not isinstance(trial, numbers.Integral) or trial < 0:
            raise ValueError('trial ID must be a non-negative integer')
        key = (int(person), int(trial))
        if key in seen:
            raise ValueError('duplicate participant/trial')
        seen.add(key)
        if split is None:
            unassigned += 1
        else:
            event_counts[split] += 1
            split_people[split].add(int(person))
    return {
        **{key: {'participants': len(split_people[key]), 'events': event_counts[key]}
           for key in ('train', 'validation')},
        'unassigned_events': unassigned,
        'coverage_pass': bool(split_people['train'] and split_people['validation']),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = Path('docs/pretrained_reuse/R2_ADL_PARTICIPANT_SPLIT_PROTOCOL_20261027.md')
    if protocol_digest(protocol) != PROTOCOL_SHA:
        raise ValueError('preregistered protocol hash mismatch')
    import h5py

    results, groups_seen, names_seen = [], {'train': set(), 'validation': set()}, set()
    for path in args.archives:
        receipt = json.loads((path.parent / 'result.json').read_text(encoding='utf-8'))
        if receipt['status'] != 'PASS_METADATA_ONLY' or receipt['archive'] != path.name:
            raise ValueError('archive preflight receipt mismatch')
        with path.open('rb') as f:
            archive_sha = hashlib.file_digest(f, 'sha256').hexdigest()
        if archive_sha != receipt['archive_sha256'] or path.name in names_seen:
            raise ValueError('archive hash mismatch or repeated activity')
        names_seen.add(path.name)
        reference = None
        checks = []
        with zipfile.ZipFile(path) as z:
            members = sorted((i for i in z.infolist() if i.filename.endswith('.hdf5')),
                             key=lambda i: i.filename)
            if len(members) != 5 or len({i.filename for i in members}) != 5:
                raise ValueError('expected five distinct sensors')
            for item in members:
                if item.file_size > 128 * 2**20 or item.flag_bits & 1:
                    raise ValueError('oversized/encrypted member')
                with h5py.File(io.BytesIO(z.read(item)), 'r') as h:
                    rows = (h['participant'][:], h['trial'][:], h['timestamp'][:])
                if reference is None:
                    reference = rows
                alignment = compare_sensor_rows(reference, rows)
                checks.append(alignment)
                if alignment['reference_only_pairs'] or alignment['sensor_only_pairs']:
                    raise ValueError('participant/trial identities differ across sensors')
        if reference is None or not np.isfinite(reference[2]).all():
            raise ValueError('empty or nonfinite activity metadata')
        summary = summarize_activity(reference[0], reference[1])
        for person in set(map(int, reference[0])):
            split = participant_split(person)
            if split:
                groups_seen[split].add(person)
        results.append({'archive': path.name, 'archive_sha256': archive_sha,
                        'summary': summary, 'sensor_checks': checks})
    if groups_seen['train'] & groups_seen['validation']:
        raise ValueError('participant appears in both splits')
    result = {'protocol_sha256': PROTOCOL_SHA, 'archives': results,
              'distinct_train_participants': len(groups_seen['train']),
              'distinct_validation_participants': len(groups_seen['validation']),
              'participant_overlap': 0, 'all_inspected_activities_coverage_pass': all(
                  row['summary']['coverage_pass'] for row in results),
              'scope': 'partial source feasibility; no fitting or validation scoring',
              'sealed_audit_loaded': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
