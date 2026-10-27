"""Remove reversible transcript-derived IDs from the fixed GRID catalog."""

import argparse
import json
from pathlib import Path

from scripts.audit_droid_train_temporal import digest
from scripts.audit_grid_quartet_distances import PINS, ROOT

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / 'docs/pretrained_reuse/R2_GRID_OPTION_ID_REPAIR_PROTOCOL_20261028.md'


def repair(row):
    codes, old_ids, queries = row['codes'], row['option_ids'], row['queries']
    if (len(codes) != 4 or len(set(codes)) != 4 or len(old_ids) != 4
            or len(set(old_ids)) != 4 or len(queries) != 4):
        raise ValueError('invalid original quartet')
    old_pairs = {(c, i) for c, i in zip(codes, old_ids, strict=True)}
    pairs = {(q['audio_code'], q['target_id']) for q in queries}
    if len(pairs) != 4 or pairs != old_pairs:
        raise ValueError('invalid original target mapping')
    new_ids = [f'option-{i}' for i in range(4)]
    by_code = dict(zip(codes, new_ids, strict=True))
    output = dict(row)
    output['option_ids'] = new_ids
    output['queries'] = [{'audio_code': q['audio_code'], 'target_id': by_code[q['audio_code']]}
                         for q in queries]
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    out = parser.parse_args().output_dir.resolve()
    if out.exists() or out.is_relative_to(REPO):
        raise ValueError('new external output required')
    if any(digest(ROOT / f'{p}.jsonl') != h for p, h in PINS.items()):
        raise ValueError('source hash mismatch')
    out.mkdir(parents=True)
    summary = {}
    for part in PINS:
        source, output = ROOT / f'{part}.jsonl', out / f'{part}.jsonl'
        count = 0
        with (source.open(encoding='utf-8') as inp,
              output.open('x', encoding='utf-8', newline='\n') as dst):
            for line in inp:
                row = repair(json.loads(line))
                dst.write(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n')
                count += 1
        if digest(source) != PINS[part]:
            raise ValueError('source changed')
        summary[part] = {'quartets': count, 'roles': 4 * count, 'sha256': digest(output)}
    result = {'counts': summary, 'source_hashes': PINS, 'protocol_sha256': digest(PROTOCOL),
              'script_sha256': digest(Path(__file__)), 'media_read': False}
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
