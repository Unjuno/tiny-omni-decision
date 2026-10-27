"""Verify overlapping GRID alignment text from the original Zenodo archive."""

import argparse
import collections
import json
import re
import zipfile
from pathlib import Path

from scripts.audit_droid_train_temporal import digest
from scripts.audit_grid_alignment_content import words_match
from scripts.audit_grid_lombard_index import TRAIN, TRAIN_SHA

ARCHIVE = Path('C:/CodexArtifacts/pretrained-reuse-r2/'
               'grid-alignments-20261028-attempt-01/alignments.zip')
ARCHIVE_SHA = '161bcf2548797e6cf0116541e8c76fdb34e014a1dd4d33d2909cb12e588232ee'
REPO = Path(__file__).resolve().parents[1]
ENTRY = re.compile(r'^alignments/s\d+/([a-z0-9]{6})\.align$')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    if out.exists() or out.is_relative_to(REPO):
        raise ValueError('new external output required')
    if digest(ARCHIVE) != ARCHIVE_SHA or digest(TRAIN) != TRAIN_SHA:
        raise ValueError('pinned input mismatch')
    rows = json.loads(TRAIN.read_bytes())
    by_code = collections.defaultdict(list)
    for row in rows:
        by_code[row['actual_code']].append(row)
    seen, counts = set(), collections.Counter()
    with zipfile.ZipFile(ARCHIVE) as archive:
        for member in archive.infolist():
            match = ENTRY.fullmatch(member.filename)
            if not match or match[1] not in by_code:
                continue
            code = match[1]
            if code in seen or member.file_size > 4096:
                raise ValueError('duplicate or unexpectedly large matching entry')
            seen.add(code)
            lines = archive.read(member).decode('utf-8').splitlines()
            words = [line.split()[-1] for line in lines
                     if line.split() and line.split()[-1] not in ('sil', 'sp')]
            for row in by_code[code]:
                counts['comparisons'] += 1
                counts['exact'] += int(words_match(words, row['targets']))
    result = {'overlap_codes': len(seen), 'comparisons': counts['comparisons'],
              'exact_word_matches': counts['exact'], 'archive_sha256': ARCHIVE_SHA,
              'train_sha256': TRAIN_SHA, 'script_sha256': digest(Path(__file__)),
              'media_read': False}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
