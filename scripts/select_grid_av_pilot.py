"""Freeze a small, speaker-balanced GRID AV pilot before reading media or predictions."""

import argparse
import hashlib
import json
from pathlib import Path

from scripts.audit_droid_train_temporal import digest

ROOT = Path('C:/CodexArtifacts/pretrained-reuse-r2')
SOURCE = ROOT / 'grid-two-slot-correspondence-20261028-attempt-01/validation.jsonl'
SOURCE_SHA256 = 'ff7ec51c8bf8e873c326d41771a822b400ec99a3a8d5ee6871ad92647a364aa2'
PROTOCOL = Path('docs/pretrained_reuse/R0_TASK_DEVICE_AND_GRID_AV_GATE_20261028.md')
SPEAKERS = ('s30', 's4', 's25', 's13', 's24', 's23', 's18', 's27')
REPO = Path(__file__).resolve().parents[1]


def _valid_row(row):
    if not isinstance(row, dict):
        raise ValueError('quartet row must be an object')
    speaker, quartet_id = row.get('speaker'), row.get('quartet_id')
    codes, option_ids, queries = row.get('codes'), row.get('option_ids'), row.get('queries')
    if (not isinstance(speaker, str) or not speaker
            or not isinstance(quartet_id, str) or len(quartet_id) != 64
            or any(c not in '0123456789abcdef' for c in quartet_id)
            or not isinstance(codes, list) or len(codes) != 4
            or any(not isinstance(c, str) or not c for c in codes) or len(set(codes)) != 4
            or not isinstance(option_ids, list) or len(option_ids) != 4
            or any(not isinstance(i, str) or not i for i in option_ids)
            or len(set(option_ids)) != 4 or not isinstance(queries, list) or len(queries) != 4):
        raise ValueError('invalid quartet identity or candidate structure')
    expected = set(zip(codes, option_ids, strict=True))
    pairs = []
    for query in queries:
        if (not isinstance(query, dict) or not isinstance(query.get('audio_code'), str)
                or not isinstance(query.get('target_id'), str)):
            raise ValueError('invalid query role')
        pairs.append((query['audio_code'], query['target_id']))
    if len(set(pairs)) != 4 or set(pairs) != expected or {target for _, target in pairs} != set(
        option_ids
    ):
        raise ValueError('quartet roles are not balanced')


def select_pilot(rows, speakers=SPEAKERS, per_speaker=8):
    """Return the fixed lowest-hash quartets per speaker, independent of row order."""
    speakers = tuple(speakers)
    if (not speakers or len(set(speakers)) != len(speakers)
            or not isinstance(per_speaker, int) or isinstance(per_speaker, bool)
            or per_speaker < 1):
        raise ValueError('invalid speaker selection or per-speaker count')
    grouped = {speaker: [] for speaker in speakers}
    quartet_ids = set()
    for row in rows:
        _valid_row(row)
        speaker, quartet_id = row['speaker'], row['quartet_id']
        if speaker not in grouped:
            raise ValueError('unexpected speaker in validation manifest')
        if quartet_id in quartet_ids:
            raise ValueError('duplicate quartet ID')
        quartet_ids.add(quartet_id)
        grouped[speaker].append(row)
    if any(len(grouped[speaker]) < per_speaker for speaker in speakers):
        raise ValueError('validation manifest is missing a required speaker or quartets')
    selected = []
    for speaker in speakers:
        ranked = sorted(
            grouped[speaker],
            key=lambda row: (
                hashlib.sha256(f"grid-av-pilot-v1|{row['quartet_id']}".encode('ascii')).hexdigest(),
                row['quartet_id'],
            ),
        )
        selected.extend(ranked[:per_speaker])
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() or out.is_relative_to(REPO):
        raise ValueError('use a new external output directory')
    if digest(SOURCE) != SOURCE_SHA256:
        raise ValueError('pinned GRID validation manifest hash mismatch')
    rows = [json.loads(line) for line in SOURCE.read_text(encoding='utf-8').splitlines()]
    selected = select_pilot(rows)
    if digest(SOURCE) != SOURCE_SHA256:
        raise ValueError('GRID validation manifest changed during selection')
    out.mkdir(parents=True)
    path = out / 'selected.jsonl'
    with path.open('x', encoding='utf-8', newline='\n') as file:
        for row in selected:
            file.write(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n')
    summary = {
        'scope': 'pre-media development pilot selection; no predictions read',
        'source_path': str(SOURCE),
        'source_sha256': SOURCE_SHA256,
        'protocol_sha256': digest(PROTOCOL),
        'script_sha256': digest(Path(__file__)),
        'sample_sha256': digest(path),
        'speakers': list(SPEAKERS),
        'quartets_per_speaker': 8,
        'quartets': len(selected),
        'query_roles': sum(len(row['queries']) for row in selected),
        'unique_audio_codes': len({q['audio_code'] for row in selected for q in row['queries']}),
        'unique_video_codes': len({code for row in selected for code in row['codes']}),
        'media_read': False,
        'predictions_read': False,
        'sealed_audit_read': False,
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
