"""Compare a pinned derivative GRID index with existing Lombard train codes."""

import argparse
import hashlib
import json
from pathlib import Path

import requests

from scripts.audit_droid_train_temporal import digest

URL = ('https://huggingface.co/datasets/SilentSpeak/EGCLLC/resolve/'
       'b61192d1a5790843425862111a0eb870d911e8a9/alignments.txt')
INDEX_SHA = '99f9abfc6e01bd922dab0049357ee23de106aef34d26a0c08edef83511eb08be'
TRAIN = Path('C:/CodexArtifacts/pretrained-reuse-r2/lombard-grid-layout-02-media-gated/train.json')
TRAIN_SHA = '259f3ee36aa956618a89819338c2ecb67b194c82dbf636df70b551e00a5d78b0'
REPO = Path(__file__).resolve().parents[1]


def compare(lines, rows):
    if any('/align/' not in s for s in lines) or len(set(lines)) != len(lines):
        raise ValueError('invalid or duplicate index entries')
    codes = {line.rsplit('/', 1)[-1] for line in lines}
    if any(len(c) != 6 or not c.isalnum() for c in codes):
        raise ValueError('unexpected utterance code')
    train_codes = {row['actual_code'] for row in rows}
    return {'index_rows': len(lines), 'index_unique_codes': len(codes),
            'train_rows': len(rows), 'train_unique_codes': len(train_codes),
            'overlap_unique_codes': len(codes & train_codes),
            'overlap_train_rows': sum(row['actual_code'] in codes for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    if out.exists() or out.is_relative_to(REPO) or digest(TRAIN) != TRAIN_SHA:
        raise ValueError('invalid output or train hash')
    response = requests.get(URL, timeout=30)
    response.raise_for_status()
    index_hash = hashlib.sha256(response.content).hexdigest()
    if len(response.content) > 2_000_000 or index_hash != INDEX_SHA:
        raise ValueError('index hash mismatch')
    rows = json.loads(TRAIN.read_bytes())
    result = {'counts': compare(response.text.splitlines(), rows), 'index_url': URL,
              'index_sha256': INDEX_SHA, 'lombard_train_sha256': TRAIN_SHA,
              'script_sha256': digest(Path(__file__)), 'media_downloaded': False}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
