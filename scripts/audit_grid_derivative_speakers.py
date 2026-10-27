"""Compare speaker labels in GRID's original and pinned derivative indices."""

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

import requests

from scripts.audit_droid_train_temporal import digest
from scripts.audit_grid_lombard_index import INDEX_SHA, URL
from scripts.audit_grid_original_content import ARCHIVE, ARCHIVE_SHA

REPO = Path(__file__).resolve().parents[1]
ORIGINAL = re.compile(r'^alignments/(s\d+)/([a-z0-9]{6})\.align$')
DERIVATIVE = re.compile(r'^GRID_alignments/(s\d+)/align/([a-z0-9]{6})$')


def parse_unique(names, pattern):
    result = {}
    for name in names:
        match = pattern.fullmatch(name)
        if not match:
            continue
        speaker, code = match.groups()
        if code in result:
            raise ValueError('duplicate utterance code')
        result[code] = speaker
    return result


def compare(original, derivative):
    shared = set(original) & set(derivative)
    pairs = {(original[c], derivative[c]) for c in shared}
    return {'original_codes': len(original), 'derivative_codes': len(derivative),
            'shared_codes': len(shared), 'original_only_codes': len(set(original) - shared),
            'derivative_only_codes': len(set(derivative) - shared),
            'shared_different_speaker_labels': sum(original[c] != derivative[c] for c in shared),
            'speaker_pairs': len(pairs), 'original_speakers': len({a for a, _ in pairs}),
            'derivative_speakers': len({b for _, b in pairs}),
            'equal_speaker_labels': sum(a == b for a, b in pairs),
            'different_speaker_labels': sum(a != b for a, b in pairs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    if out.exists() or out.is_relative_to(REPO) or digest(ARCHIVE) != ARCHIVE_SHA:
        raise ValueError('output or archive hash mismatch')
    response = requests.get(URL, timeout=30)
    response.raise_for_status()
    if hashlib.sha256(response.content).hexdigest() != INDEX_SHA:
        raise ValueError('derivative index hash mismatch')
    with zipfile.ZipFile(ARCHIVE) as archive:
        original = parse_unique(archive.namelist(), ORIGINAL)
    derivative = parse_unique(response.text.splitlines(), DERIVATIVE)
    if len(original) != 34000 or len(derivative) != 32827:
        raise ValueError('unexpected source coverage')
    result = {'counts': compare(original, derivative), 'original_sha256': ARCHIVE_SHA,
              'derivative_sha256': INDEX_SHA, 'script_sha256': digest(Path(__file__)),
              'speaker_identity_verified': False, 'media_read': False}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as file:
        json.dump(result, file, indent=2)
        file.write('\n')


if __name__ == '__main__':
    main()
