"""Fix the new native AV command layout, retaining unresolved labels separately."""

import argparse
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from scripts.lombard_grid_corpus import (
    SLOTS,
    decode_code,
    parse_member,
    partition_groups,
    reconcile_label,
    speaker_groups,
)

ROOT = Path('C:/CodexArtifacts/pretrained-reuse-r2')
PINS = {
    'lombard-grid-admission-01/lombardgrid_json.zip':
        '1de3299baaa6a13d8bcd458a688c94f2288a0ecca8b5a6817a055b393f5a98a6',
    'lombard-grid-archive-index-01/audio-members.json':
        '8ed96b356ff9bdfadc29f3cce7cb58c00d5577946a1c71cf02607f93744e510b',
    'lombard-grid-archive-index-01/front-members.json':
        '337cfcfb4df687db7a04c3c4fe8a3589b8f9259d716560b5f0503fc41d6a118b',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for name, expected in PINS.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError('pinned source snapshot differs')
    metadata = defaultdict(list)
    with zipfile.ZipFile(ROOT / next(iter(PINS))) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            for row in json.loads(archive.read(member)):
                metadata[row['SPKR'], row['COND'], row['UTTERANCE']].append(row)
    indices = {}
    for kind in ('audio', 'front'):
        records = json.loads((ROOT / f'lombard-grid-archive-index-01/{kind}-members.json')
                             .read_text(encoding='utf-8'))
        indices[kind] = {PurePosixPath(r['name']).stem: r for r in records}
        if len(indices[kind]) != len(records) or len(records) != 5390:
            raise ValueError('duplicate/changed archive asset count')
    if indices['audio'].keys() != indices['front'].keys():
        raise ValueError('audio/front media identities differ')
    rows = []
    for stem, asset in sorted(indices['audio'].items()):
        row = parse_member(asset['name'])
        matches = metadata[row['speaker'], row['condition'], row['intended_code']]
        row = reconcile_label(row, matches)
        row['assets'] = {kind: indices[kind][stem] for kind in indices}
        rows.append(row)
    groups, assignments = speaker_groups(rows), partition_groups(rows)
    splits = {name: [] for name in ('train', 'calibration', 'validation')}
    unresolved, phrases = [], defaultdict(set)
    for row in rows:
        row.update(split=assignments[row['speaker']], speaker_group=groups[row['speaker']])
        for code in (row['intended_code'], row['actual_code']):
            try:
                decode_code(code)
            except ValueError:
                continue
            phrases[code].add(row['split'])
        if row['targets'] is None:
            unresolved.append(row)
        else:
            splits[row['split']].append(row)
    if any(len(parts) > 1 for parts in phrases.values()):
        raise ValueError('normalized complete phrase overlap')
    if any(len({r['speaker'] for r in split}) < 4 for split in splits.values()):
        raise ValueError('fixed split lacks minimum speaker coverage')
    output = args.output
    output.mkdir(exist_ok=False)
    hashes = {}
    for name, split in {**splits, 'unresolved': unresolved}.items():
        raw = json.dumps(split, sort_keys=True, indent=2).encode()
        (output / (name + '.json')).write_bytes(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    report = dict(
        status='LAYOUT_FIXED_MEDIA_INTEGRITY_PENDING', seed=17, source_pins=PINS,
        split_rule='SHA256(seed:sorted connected speakers), first8bytes modulo100; '
                   '<70 train,70..84 calibration,85..99 validation',
        grouping='speaker plus all valid intended/actual complete sentences; '
                 'published6/29,25/26 aliases',
        slots={slot: list(values.values()) for slot, values in SLOTS.items()},
        source_assets=len(rows), labeled_assets=sum(map(len, splits.values())),
        labeled_coverage=sum(map(len, splits.values())) / len(rows),
        unresolved_assets=len(unresolved),
        unresolved_counts=dict(Counter(r['label_status'] for r in unresolved)),
        metadata_missing_assets=sum(not r['metadata_records'] for r in rows),
        label_policy='exact source actual code; quarantine any valid metadata disagreement; '
                     'no cleanup of error/silence/distorted markers; filename-only labels '
                     'explicit when metadata missing; no model-informed eligibility',
        speaker_groups=len(set(groups.values())), speaker_assignments=assignments,
        splits={s: dict(assets=len(r), speakers=len({v['speaker'] for v in r}),
                        conditions=dict(Counter(v['condition'] for v in r)),
                        slot_counts={slot: dict(Counter(v['targets'][slot] for v in r))
                                     for slot in SLOTS}) for s, r in splits.items()},
        hashes=hashes, known_source_speaker_phrase_cross_split_overlap=0,
        assets_acquired=False, decoded_duplicate_check='PENDING',
        precise_av_synchronization='PENDING', quality_evaluated=False,
        runtime_input_contract='waveform and video pixels/timestamps only; never filenames, '
                               'metadata, transcripts or alignment as input features',
        limitations='native command-slot decisions only; not general world reasoning; '
                    'unresolved assets retained and coverage reported, not scored as correct',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(Path(__file__).with_name('lombard_grid_corpus.py')
                                    .read_bytes()).hexdigest(),
    )
    (output / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['status', 'source_assets', 'labeled_assets',
                                           'unresolved_assets', 'unresolved_counts',
                                           'speaker_groups']}))
    print(json.dumps({s: {k: v for k, v in r.items() if k != 'slot_counts'}
                      for s, r in report['splits'].items()}))


if __name__ == '__main__':
    main()
