"""Screen pinned OSSL-v2 metadata; do not fetch media or split lists."""

import argparse
import collections
import csv
import hashlib
import io
import json
from pathlib import Path

import requests

REVISION = '3ed5a37d933084e5ec134e2138b8faa156cf021b'
EXPECTED = '0e7bdd7b92c29e9117a635ebe40d29ed6d82d04c60df5b3412e1d0bc728b4cbf'
URL = f'https://huggingface.co/datasets/McAuley-Lab/OSSL-v2/resolve/{REVISION}/metadata.csv'
REPO = Path(__file__).resolve().parents[1]


def summarize(rows):
    splits = {row['split'] for row in rows}
    if splits != {'train', 'test'} or len({r['movie_id'] for r in rows}) != len(rows):
        raise ValueError('unexpected split or duplicate movie ID')
    train = [row for row in rows if row['split'] == 'train']
    test = [row for row in rows if row['split'] == 'test']
    train_ids = collections.Counter(r['youtube_id'] for r in train if r['youtube_id'])
    test_ids = {r['youtube_id'] for r in test if r['youtube_id']}
    return {'movie_rows': len(rows), 'train_films': len(train),
            'train_clips': sum(int(r['n_clips']) for r in train),
            'train_unique_youtube_ids': len(train_ids),
            'train_duplicate_source_ids': sum(v > 1 for v in train_ids.values()),
            'train_rows_with_repeated_source_id': sum(v for v in train_ids.values() if v > 1),
            'train_test_shared_source_ids': len(set(train_ids) & test_ids),
            'train_missing_youtube_id': sum(not r['youtube_id'] for r in train)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output.resolve()
    if output.exists() or output.is_relative_to(REPO):
        raise ValueError('new external output required')
    response = requests.get(URL, timeout=30)
    response.raise_for_status()
    payload = response.content
    if len(payload) > 10_000_000 or hashlib.sha256(payload).hexdigest() != EXPECTED:
        raise ValueError('unexpected metadata bytes')
    rows = list(csv.DictReader(io.StringIO(payload.decode('utf-8-sig'))))
    result = {'revision': REVISION, 'metadata_sha256': EXPECTED,
              'metadata_bytes': len(payload), 'counts': summarize(rows),
              'media_fetched': False, 'admission': 'REVIEW'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
