"""Compare overlapping derivative GRID alignment words with Lombard train labels."""

import argparse
import collections
import json
import tarfile
from pathlib import Path

from scripts.audit_droid_train_temporal import digest
from scripts.audit_grid_lombard_index import TRAIN, TRAIN_SHA
from scripts.fetch_grid_derivative_alignments import SHA

ARCHIVE = Path('C:/CodexArtifacts/pretrained-reuse-r2/'
               'grid-derivative-alignments-20261028-attempt-01/alignments.tar')
REPO = Path(__file__).resolve().parents[1]
NUMBERS = dict(zip('0123456789', 'zero one two three four five six seven eight nine'.split(),
                   strict=True))


def words_match(words, targets):
    number = str(targets['number'])
    expected = [targets['command'], targets['color'], targets['preposition'],
                targets['letter'], NUMBERS.get(number, number), targets['adverb']]
    return words == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    if out.exists() or out.is_relative_to(REPO):
        raise ValueError('new external output required')
    if digest(ARCHIVE) != SHA or digest(TRAIN) != TRAIN_SHA:
        raise ValueError('pinned input mismatch')
    rows = json.loads(TRAIN.read_bytes())
    by_code = collections.defaultdict(list)
    for row in rows:
        by_code[row['actual_code']].append(row)
    seen, counts = set(), collections.Counter()
    with tarfile.open(ARCHIVE) as archive:
        for member in archive:
            if not member.isfile():
                continue
            if not member.name.endswith('.align') or member.size > 4096:
                raise ValueError('unexpected alignment entry')
            code = Path(member.name).stem
            if code not in by_code:
                continue
            if code in seen:
                raise ValueError('duplicate overlapping code')
            seen.add(code)
            lines = archive.extractfile(member).read().decode('utf-8').splitlines()
            words = [line.split()[-1] for line in lines
                     if line.split() and line.split()[-1] not in ('sil', 'sp')]
            for row in by_code[code]:
                counts['comparisons'] += 1
                counts['exact'] += int(words_match(words, row['targets']))
    result = {'overlap_codes': len(seen), 'comparisons': counts['comparisons'],
              'exact_word_matches': counts['exact'], 'archive_sha256': SHA,
              'train_sha256': TRAIN_SHA, 'script_sha256': digest(Path(__file__)),
              'media_read': False}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
