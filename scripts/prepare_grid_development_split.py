"""Build the fixed original GRID development split from alignments only."""

import argparse
import collections
import hashlib
import json
import zipfile
from pathlib import Path

from scripts.audit_droid_train_temporal import digest
from scripts.audit_grid_alignment_content import words_match
from scripts.audit_grid_original_content import ARCHIVE, ARCHIVE_SHA
from scripts.lombard_grid_corpus import decode_code

ROOT = Path('C:/CodexArtifacts/pretrained-reuse-r2/lombard-grid-layout-02-media-gated')
PINS = {
    ROOT / 'train.json': '259f3ee36aa956618a89819338c2ecb67b194c82dbf636df70b551e00a5d78b0',
    ROOT / 'calibration.json': '22748b0942dfb1e54dcbd1e356b5e25930da46f95f15724a5f0bfd03b546a765',
    ROOT / 'validation.json': '98d81114e73dd4924c087df08873d21648a2ae34a9cc289887eb0767ded4658d',
    ARCHIVE: ARCHIVE_SHA,
}
REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / 'docs/pretrained_reuse/R2_GRID_DEVELOPMENT_SPLIT_PROTOCOL_20261028.md'


def allocate(speakers):
    if len(speakers) != 32 or len(set(speakers)) != 32:
        raise ValueError('expected 32 distinct speakers')
    ordered = sorted(speakers, key=lambda s: hashlib.sha256(
        f'grid-development-v1|{s}'.encode('ascii')).hexdigest())
    return {'train': ordered[:24], 'validation': ordered[24:]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    out = parser.parse_args().output_dir.resolve()
    if out.exists() or out.is_relative_to(REPO):
        raise ValueError('new external output directory required')
    if any(digest(p) != h for p, h in PINS.items()):
        raise ValueError('pinned input mismatch')
    forbidden = set()
    for path in PINS:
        if path.suffix == '.json':
            forbidden.update(row['actual_code'] for row in json.loads(path.read_bytes())
                             if row.get('actual_code'))
    speakers = [f's{i}' for i in range(1, 35) if i not in (1, 21)]
    split = allocate(speakers)
    owner = {speaker: name for name, values in split.items() for speaker in values}
    rows = {'train': [], 'validation': []}
    seen, counts = set(), collections.Counter()
    with zipfile.ZipFile(ARCHIVE) as archive:
        for item in archive.infolist():
            parts = Path(item.filename).parts
            if (len(parts) != 3 or parts[0] != 'alignments' or parts[1] not in owner
                    or not item.filename.endswith('.align')):
                continue
            code = Path(parts[2]).stem
            if code in seen:
                raise ValueError('duplicate code')
            seen.add(code)
            target = decode_code(code)
            if code in forbidden:
                counts['excluded_seen_code'] += 1
                continue
            if item.file_size > 4096:
                raise ValueError('oversized alignment')
            lines = archive.read(item).decode('utf-8').splitlines()
            words = [line.split()[-1] for line in lines
                     if line.split() and line.split()[-1] not in ('sil', 'sp')]
            if not words_match(words, target):
                raise ValueError('alignment and decoded code differ')
            rows[owner[parts[1]]].append({'speaker': parts[1], 'code': code, 'targets': target})
    for part, items in rows.items():
        items.sort(key=lambda row: (row['speaker'], row['code']))
        if {r['speaker'] for r in items} != set(split[part]):
            raise ValueError('missing source speaker')
        if any(r['code'] in forbidden for r in items):
            raise ValueError('known content overlap')
    if {r['code'] for r in rows['train']} & {r['code'] for r in rows['validation']}:
        raise ValueError('cross-split content overlap')
    if any(digest(p) != h for p, h in PINS.items()):
        raise ValueError('input changed during preparation')
    out.mkdir(parents=True)
    for part, items in rows.items():
        encoded = json.dumps(items, sort_keys=True) + '\n'
        (out / f'{part}.json').write_text(encoded, encoding='utf-8')
    result = {'scope': 'provisional development; no media verified', 'speakers': split,
              'counts': {'train': len(rows['train']), 'validation': len(rows['validation']),
                         'excluded_seen_codes': counts['excluded_seen_code'],
                         'lombard_forbidden_unique': len(forbidden)},
              'input_hashes': {str(p): h for p, h in PINS.items()},
              'output_hashes': {part: digest(out / f'{part}.json') for part in rows},
              'protocol_sha256': digest(PROTOCOL), 'script_sha256': digest(Path(__file__)),
              'media_read': False, 'independent_persons_verified': False}
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
