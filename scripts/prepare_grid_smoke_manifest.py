"""Fix the original GRID s1 input-smoke IDs before reading any media."""

import argparse
import json
import re
import zipfile
from pathlib import Path

from scripts.audit_droid_train_temporal import digest
from scripts.audit_grid_alignment_content import words_match
from scripts.audit_grid_lombard_index import TRAIN, TRAIN_SHA
from scripts.audit_grid_original_content import ARCHIVE, ARCHIVE_SHA
from scripts.lombard_grid_corpus import decode_code

REPO = Path(__file__).resolve().parents[1]
ENTRY = re.compile(r'^alignments/s1/([a-z0-9]{6})\.align$')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    if out.exists() or out.is_relative_to(REPO):
        raise ValueError('new external output required')
    if digest(TRAIN) != TRAIN_SHA or digest(ARCHIVE) != ARCHIVE_SHA:
        raise ValueError('pinned source mismatch')
    forbidden = {row['actual_code'] for row in json.loads(TRAIN.read_bytes())}
    with zipfile.ZipFile(ARCHIVE) as archive:
        entries = [(m.group(1), item) for item in archive.infolist()
                   if (m := ENTRY.fullmatch(item.filename))]
        if len({code for code, _ in entries}) != len(entries):
            raise ValueError('duplicate s1 utterance code')
        selected = sorted((code, item) for code, item in entries if code not in forbidden)[:16]
        if len(selected) != 16:
            raise ValueError('insufficient eligible alignments')
        result_rows = []
        for code, item in selected:
            if item.file_size > 4096:
                raise ValueError('oversized alignment')
            lines = archive.read(item).decode('utf-8').splitlines()
            words = [line.split()[-1] for line in lines
                     if line.split() and line.split()[-1] not in ('sil', 'sp')]
            targets = decode_code(code)
            if not words_match(words, targets):
                raise ValueError('alignment words disagree with code')
            result_rows.append({'speaker': 's1', 'code': code, 'targets': targets})
    result = {'scope': 'technical smoke only', 'rows': result_rows,
              'selection': 'first 16 lexicographic s1 codes not in Lombard train',
              'alignment_sha256': ARCHIVE_SHA, 'lombard_train_sha256': TRAIN_SHA,
              'script_sha256': digest(Path(__file__)), 'media_read': False}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as file:
        json.dump(result, file, indent=2)
        file.write('\n')


if __name__ == '__main__':
    main()
