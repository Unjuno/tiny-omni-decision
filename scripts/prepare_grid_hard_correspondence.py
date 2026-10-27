"""Construct balanced hard GRID correspondence quartets from fixed metadata."""

import argparse
import collections
import hashlib
import json
from pathlib import Path

from scripts.audit_droid_train_temporal import digest

ROOT = Path('C:/CodexArtifacts/pretrained-reuse-r2/grid-development-split-20261028-attempt-02')
PINS = {'train': 'f6f045931aee53e576fca22efda0bc888a983aed685a30edc38b2918ff156817',
        'validation': 'c1e77098327841fadc283bb14813c38de94e07fd16728d9cc3602bc261961e11'}
REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / 'docs/pretrained_reuse/R2_GRID_HARD_CORRESPONDENCE_PROTOCOL_20261028.md'


def sha(text):
    return hashlib.sha256(text.encode('ascii')).hexdigest()


def patterns(code):
    if len(code) != 6 or not code.isalnum():
        raise ValueError('invalid code')
    return [code[:i] + '*' + code[i + 1:] for i in range(6)]


def construct(rows):
    speakers = collections.defaultdict(list)
    ids = set()
    for row in rows:
        code, speaker = row['code'], row['speaker']
        if code in ids:
            raise ValueError('duplicate code')
        ids.add(code)
        speakers[speaker].append(code)
    quartets, skipped = [], 0
    for speaker in sorted(speakers):
        codes = sorted(speakers[speaker])
        index = collections.defaultdict(list)
        for code in codes:
            for pattern in patterns(code):
                index[pattern].append(code)
        seen = set()
        for anchor in codes:
            candidates = {other for pattern in patterns(anchor) for other in index[pattern]
                          if other != anchor}
            ranked = sorted(candidates, key=lambda c: sha(f'grid-hard-v1|{anchor}|{c}'))
            if len(ranked) < 3:
                skipped += 1
                continue
            chosen = (anchor, *ranked[:3])
            if any(sum(a != b for a, b in zip(anchor, c, strict=True)) != 1
                   for c in chosen[1:]):
                raise ValueError('hard distance mismatch')
            canonical = tuple(sorted(chosen))
            if canonical in seen:
                continue
            seen.add(canonical)
            order = sorted(canonical, key=lambda c: sha(f'grid-candidate-v1|{c}'))
            opaque = [sha(f'grid-candidate-v1|{c}') for c in order]
            quartet_id = sha('grid-quartet-v1|' + '|'.join(canonical))
            quartets.append({'speaker': speaker, 'quartet_id': quartet_id, 'anchor': anchor,
                             'codes': order, 'option_ids': opaque,
                             'queries': [{'audio_code': code, 'target_id': opaque[i]}
                                         for i, code in enumerate(order)]})
    return quartets, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    out = parser.parse_args().output_dir.resolve()
    if out.exists() or out.is_relative_to(REPO):
        raise ValueError('new external output directory required')
    if any(digest(ROOT / f'{part}.json') != expected for part, expected in PINS.items()):
        raise ValueError('source split hash mismatch')
    rows = {part: json.loads((ROOT / f'{part}.json').read_bytes()) for part in PINS}
    for key in ('code', 'speaker'):
        if {row[key] for row in rows['train']} & {row[key] for row in rows['validation']}:
            raise ValueError('cross-split overlap')
    result = {}
    for part, examples in rows.items():
        quartets, skipped = construct(examples)
        if not quartets:
            raise ValueError('empty quartet set')
        result[part] = {'quartets': quartets, 'anchors_without_three_neighbors': skipped}
    if any(digest(ROOT / f'{part}.json') != expected for part, expected in PINS.items()):
        raise ValueError('source changed during generation')
    out.mkdir(parents=True)
    summary = {}
    for part, group in result.items():
        path = out / f'{part}.jsonl'
        with path.open('x', encoding='utf-8', newline='\n') as file:
            for row in group['quartets']:
                file.write(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n')
        skipped = group['anchors_without_three_neighbors']
        summary[part] = {'quartets': len(group['quartets']),
                         'query_roles': 4 * len(group['quartets']),
                         'anchors_without_three_neighbors': skipped,
                         'sha256': digest(path)}
    receipt = {'counts': summary, 'source_hashes': PINS, 'protocol_sha256': digest(PROTOCOL),
               'script_sha256': digest(Path(__file__)), 'media_read': False,
               'independent_people_verified': False}
    (out / 'summary.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
