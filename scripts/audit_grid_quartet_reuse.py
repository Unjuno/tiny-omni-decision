"""Count speaker and audio reuse in fixed GRID correspondence catalogs."""

import argparse
import collections
import json
from pathlib import Path

from scripts.audit_droid_train_temporal import digest
from scripts.audit_grid_quartet_distances import PINS, ROOT

REPO = Path(__file__).resolve().parents[1]


def summarize(rows):
    speakers, audio = collections.Counter(), collections.Counter()
    count = 0
    for row in rows:
        queries = row['queries']
        if (len(queries) != 4 or len({q['audio_code'] for q in queries}) != 4
                or {q['audio_code'] for q in queries} != set(row['codes'])):
            raise ValueError('invalid query roles')
        count += 1
        speakers[row['speaker']] += 1
        audio.update(q['audio_code'] for q in queries)
    if not count:
        raise ValueError('empty corpus')
    return {'quartets': count, 'speaker_labels': len(speakers), 'unique_audio_codes': len(audio),
            'min_quartets_per_speaker': min(speakers.values()),
            'max_quartets_per_speaker': max(speakers.values()),
            'min_roles_per_audio': min(audio.values()), 'max_roles_per_audio': max(audio.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    if out.exists() or out.is_relative_to(REPO):
        raise ValueError('new external output required')
    if any(digest(ROOT / f'{p}.jsonl') != h for p, h in PINS.items()):
        raise ValueError('source hash mismatch')
    counts = {}
    for part in PINS:
        with (ROOT / f'{part}.jsonl').open(encoding='utf-8') as file:
            counts[part] = summarize(json.loads(line) for line in file)
    if any(digest(ROOT / f'{p}.jsonl') != h for p, h in PINS.items()):
        raise ValueError('source changed')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as file:
        json.dump({'counts': counts, 'source_hashes': PINS,
                   'script_sha256': digest(Path(__file__)), 'media_read': False}, file, indent=2)
        file.write('\n')


if __name__ == '__main__':
    main()
