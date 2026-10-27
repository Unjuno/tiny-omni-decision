"""Count all-role distances in the frozen GRID development quartet catalogs."""

import argparse
import collections
import json
from pathlib import Path

from scripts.audit_droid_train_temporal import digest

ROOT = (Path('C:/CodexArtifacts/pretrained-reuse-r2')
        / 'grid-two-slot-correspondence-20261028-attempt-01')
PINS = {'train': 'fc3241bc146156d24f4f92d2f8769d4ad758e7e5f97611c14fb05dfc0d5322ea',
        'validation': 'ff7ec51c8bf8e873c326d41771a822b400ec99a3a8d5ee6871ad92647a364aa2'}
REPO = Path(__file__).resolve().parents[1]


def distances(codes):
    if len(codes) != 4 or len(set(codes)) != 4 or any(len(c) != 6 for c in codes):
        raise ValueError('invalid quartet')
    return [[sum(a != b for a, b in zip(code, other, strict=True))
             for j, other in enumerate(codes) if i != j]
            for i, code in enumerate(codes)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    if out.exists() or out.is_relative_to(REPO):
        raise ValueError('new external output required')
    if any(digest(ROOT / f'{part}.jsonl') != sha for part, sha in PINS.items()):
        raise ValueError('source hash mismatch')
    result = {}
    for part in PINS:
        pair_hist, maximum_hist = collections.Counter(), collections.Counter()
        quartets = 0
        with (ROOT / f'{part}.jsonl').open(encoding='utf-8') as file:
            for line in file:
                row = json.loads(line)
                roles = distances(row['codes'])
                for values in roles:
                    pair_hist.update(values)
                    maximum_hist[max(values)] += 1
                quartets += 1
        result[part] = {'quartets': quartets, 'query_roles': 4 * quartets,
                        'ordered_query_candidate_distances': dict(sorted(pair_hist.items())),
                        'maximum_distance_per_query': dict(sorted(maximum_hist.items()))}
    if any(digest(ROOT / f'{part}.jsonl') != sha for part, sha in PINS.items()):
        raise ValueError('source changed')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as file:
        json.dump({'counts': result, 'source_hashes': PINS,
                   'script_sha256': digest(Path(__file__)), 'media_read': False}, file, indent=2)
        file.write('\n')


if __name__ == '__main__':
    main()
