"""Strict native command labels and speaker/phrase-connected corpus identities."""

import hashlib
import re
import string
from pathlib import PurePosixPath

SLOTS = {
    'command': dict(zip('blps', ['bin', 'lay', 'place', 'set'], strict=True)),
    'color': dict(zip('bgrw', ['blue', 'green', 'red', 'white'], strict=True)),
    'preposition': dict(zip('abiw', ['at', 'by', 'in', 'with'], strict=True)),
    'letter': {c: c for c in string.ascii_lowercase if c != 'w'},
    'number': {'z': 'zero', **{str(i): str(i) for i in range(1, 10)}},
    'adverb': dict(zip('anps', ['again', 'now', 'please', 'soon'], strict=True)),
}
REPEATED_RECORDING = 's14_1_50_15_r_prwr3a'


def decode_code(code):
    if (not isinstance(code, str) or len(code) != 6
            or any(c not in table for c, table in zip(code, SLOTS.values(), strict=True))):
        raise ValueError('unresolved or invalid full Grid utterance code')
    return {slot: table[c] for c, (slot, table) in zip(code, SLOTS.items(), strict=True)}


def parse_member(member):
    path = PurePosixPath(member)
    if path.suffix not in {'.wav', '.mov'}:
        raise ValueError('unexpected media suffix')
    stem = path.stem
    parts = stem.split('_')
    if stem == REPEATED_RECORDING:
        speaker, condition, intended, actual = 's14', 'p', 'prwr3a', 'prwr3a'
    elif (len(parts) in {3, 5} and re.fullmatch(r's\d+', parts[0])
          and 2 <= int(parts[0][1:]) <= 55 and parts[1] in {'p', 'l'}
          and (len(parts) == 3 or parts[3] == 'WRONG')):
        speaker, condition, intended = parts[:3]
        actual = parts[4] if len(parts) == 5 else intended
    else:
        raise ValueError('unreviewed member identity structure')
    try:
        targets = decode_code(actual)
    except ValueError:
        targets = None
    return dict(source_id=stem, speaker=speaker, condition=condition,
                intended_code=intended, actual_annotation=actual,
                actual_code=actual if targets is not None else None, targets=targets)


def speaker_groups(rows):
    speakers = {row['speaker'] for row in rows}
    parent = {speaker: speaker for speaker in speakers}

    def find(speaker):
        while parent[speaker] != speaker:
            parent[speaker] = parent[parent[speaker]]
            speaker = parent[speaker]
        return speaker

    def union(first, second):
        a, b = sorted([find(first), find(second)])
        parent[b] = a

    for first, second in [('s6', 's29'), ('s25', 's26')]:
        if first in speakers and second in speakers:
            union(first, second)
    seen = {}
    for row in rows:
        for code in [row['intended_code'], row['actual_code']]:
            try:
                decode_code(code)
            except ValueError:
                continue
            if code in seen:
                union(seen[code], row['speaker'])
            else:
                seen[code] = row['speaker']
    members = {}
    for speaker in sorted(speakers):
        members.setdefault(find(speaker), []).append(speaker)
    names = {key: '+'.join(group) for key, group in members.items()}
    return {speaker: names[find(speaker)] for speaker in sorted(speakers)}


def reconcile_label(row, metadata):
    result = dict(row)
    result['metadata_records'] = metadata
    result['label_origin'] = ('official_filename_and_available_metadata' if metadata else
                              'official_filename_only_metadata_missing')
    result['label_status'] = (
        'eligible' if row['targets'] is not None else 'unresolved_transcription'
    )
    for annotation in metadata:
        code = (annotation['TRANSCRIPTION'] if annotation['STATUS'] == 'WRONG'
                else annotation['UTTERANCE'])
        try:
            decode_code(code)
        except ValueError:
            continue
        if row['actual_code'] is not None and code != row['actual_code']:
            result.update(targets=None, label_status='metadata_conflict')
    return result


def partition_groups(rows, seed=17):
    groups = speaker_groups(rows)
    result = {}
    for speaker, group in groups.items():
        digest = hashlib.sha256(f'{seed}:{group}'.encode()).digest()
        bucket = int.from_bytes(digest[:8], 'big') % 100
        result[speaker] = 'train' if bucket < 70 else 'calibration' if bucket < 85 else 'validation'
    return result
