"""Count original GRID alignment candidates after excluding Lombard train text."""

import argparse
import collections
import json
import zipfile
from pathlib import Path

from scripts.audit_droid_train_temporal import digest
from scripts.audit_grid_lombard_index import TRAIN, TRAIN_SHA
from scripts.audit_grid_original_content import ARCHIVE, ARCHIVE_SHA, ENTRY
from scripts.lombard_grid_corpus import decode_code

REPO = Path(__file__).resolve().parents[1]


def coverage(names, forbidden):
    speakers, codes, total = collections.Counter(), set(), 0
    for name in names:
        match = ENTRY.fullmatch(name)
        if not match:
            continue
        code = match[1]
        decode_code(code)
        total += 1
        if code in codes:
            raise ValueError('original GRID code is not unique')
        codes.add(code)
        if code not in forbidden:
            speakers[name.split('/')[1]] += 1
    if not speakers:
        raise ValueError('empty candidate set')
    return {'all_alignments': total, 'overlap_codes': len(codes & forbidden),
            'eligible_alignment_rows': sum(speakers.values()),
            'speakers': len(speakers), 'min_per_speaker': min(speakers.values()),
            'max_per_speaker': max(speakers.values()),
            's21_audio_only_eligible': speakers['s21'],
            'other_speaker_eligible': sum(n for s, n in speakers.items() if s != 's21')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    if out.exists() or out.is_relative_to(REPO):
        raise ValueError('new external output required')
    if digest(ARCHIVE) != ARCHIVE_SHA or digest(TRAIN) != TRAIN_SHA:
        raise ValueError('pinned source mismatch')
    forbidden = {r['actual_code'] for r in json.loads(TRAIN.read_bytes())}
    with zipfile.ZipFile(ARCHIVE) as archive:
        counts = coverage(archive.namelist(), forbidden)
    result = {'counts': counts, 'alignment_sha256': ARCHIVE_SHA,
              'lombard_train_sha256': TRAIN_SHA, 'script_sha256': digest(Path(__file__)),
              'media_read': False, 'independent_persons_verified': False}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as file:
        json.dump(result, file, indent=2)
        file.write('\n')


if __name__ == '__main__':
    main()
