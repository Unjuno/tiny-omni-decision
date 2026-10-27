"""Freeze a corrected GRID AV pilot from verified per-talker code inventories."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from scripts.audit_grid_direct_speaker_codes import (
    audit_speaker_bundle,
    tar_codes,
    zip_codes_from_tail,
)

SPEAKERS = ('s30', 's4', 's25', 's13', 's24', 's23', 's18', 's27')
VIDEO_ZIP_SIZES = {
    's13': 407_686_627,
    's18': 381_117_685,
    's23': 416_883_805,
    's24': 449_740_738,
    's25': 440_939_778,
    's27': 425_908_152,
    's30': 381_061_085,
    's4': 491_885_378,
}
PER_SPEAKER = 8
CODE_RE = re.compile(r'^[a-z0-9]{6}$')
REPO = Path(__file__).resolve().parents[1]


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode('ascii')).hexdigest()


def hamming_distance(left: str, right: str) -> int:
    if len(left) != 6 or len(right) != 6:
        raise ValueError('GRID code must have six characters')
    return sum(a != b for a, b in zip(left, right, strict=True))


def build_candidate_quartets(speaker: str, codes: set[str]) -> tuple[list[dict], int]:
    """Build deduplicated balanced quartets using the fixed two-slot rule."""
    if not speaker or not codes or any(not CODE_RE.fullmatch(code) for code in codes):
        raise ValueError('speaker and valid six-character codes are required')
    quartets = {}
    missing_neighbor_anchors = 0
    for anchor in sorted(codes):
        neighbors = [code for code in codes if code != anchor
                     and hamming_distance(anchor, code) == 2]
        neighbors.sort(key=lambda code: (
            _hash(f'grid-two-slot-v1|{anchor}|{code}'), code,
        ))
        if len(neighbors) < 3:
            missing_neighbor_anchors += 1
            continue
        quartet_codes = tuple(sorted((anchor, *neighbors[:3])))
        if quartet_codes in quartets:
            continue
        option_ids = [
            _hash(f'grid-direct-option-v1|{speaker}|{code}')
            for code in quartet_codes
        ]
        quartet_id = _hash(f'grid-direct-quartet-v1|{speaker}|{"|".join(quartet_codes)}')
        quartets[quartet_codes] = {
            'speaker': speaker,
            'anchor': anchor,
            'codes': list(quartet_codes),
            'option_ids': option_ids,
            'quartet_id': quartet_id,
            'queries': [
                {'audio_code': code, 'target_id': option_id}
                for code, option_id in zip(quartet_codes, option_ids, strict=True)
            ],
        }
    return [quartets[key] for key in sorted(quartets)], missing_neighbor_anchors


def select_pilot(rows: list[dict], *, speakers=SPEAKERS, per_speaker=PER_SPEAKER) -> list[dict]:
    if len(set(speakers)) != len(speakers) or per_speaker < 1:
        raise ValueError('invalid pilot speaker list or size')
    grouped = {speaker: [] for speaker in speakers}
    seen = set()
    for row in rows:
        speaker = row['speaker']
        if speaker not in grouped or row['quartet_id'] in seen:
            raise ValueError('unexpected speaker or duplicate quartet')
        if (len(row['codes']) != 4 or len(set(row['codes'])) != 4
                or len(row['queries']) != 4
                or {(q['audio_code'], q['target_id']) for q in row['queries']}
                != set(zip(row['codes'], row['option_ids'], strict=True))):
            raise ValueError('quartet does not contain four balanced query roles')
        seen.add(row['quartet_id'])
        grouped[speaker].append(row)
    if any(len(grouped[speaker]) < per_speaker for speaker in speakers):
        raise ValueError('not enough eligible quartets for every speaker')
    selected = []
    for speaker in speakers:
        ranked = sorted(
            grouped[speaker],
            key=lambda row: (_hash(f'grid-av-direct-pilot-v1|{row["quartet_id"]}'),
                             row['quartet_id']),
        )
        selected.extend(ranked[:per_speaker])
    return selected


def build_from_bundles(bundle_root: Path, frozen_validation: Path) -> tuple[list[dict], dict]:
    source_audits = {}
    all_rows = []
    speaker_counts = {}
    missing_anchors = {}
    for speaker in SPEAKERS:
        audio_tar = bundle_root / f'{speaker}.tar.part'
        alignment_tar = bundle_root / f'{speaker}-align.tar'
        video_tail = bundle_root / f'{speaker}-video-tail.bin'
        video_size = VIDEO_ZIP_SIZES[speaker]
        range_start = video_size - video_tail.stat().st_size
        audit = audit_speaker_bundle(
            audio_tar=audio_tar,
            alignment_tar=alignment_tar,
            video_tail=video_tail,
            video_range_start=range_start,
            video_total_size=video_size,
            frozen_validation=frozen_validation,
            speaker=speaker,
        )
        if audit['status'] not in {
            'PASS_DIRECT_SOURCE_BUNDLE_AND_FROZEN_SPLIT',
            'OFFICIAL_BUNDLE_MATCH_FROZEN_SPLIT_MISMATCH',
        }:
            raise ValueError(f'direct source inventory mismatch for {speaker}')
        audio = tar_codes(audio_tar, '.wav')
        alignments = tar_codes(alignment_tar, '.align')
        videos, _ = zip_codes_from_tail(
            video_tail, range_start=range_start, total_size=video_size,
        )
        if audio != alignments or audio != videos:
            raise ValueError(f'direct media code inventory mismatch for {speaker}')
        rows, missing = build_candidate_quartets(speaker, audio)
        if len(rows) < PER_SPEAKER:
            raise ValueError(f'insufficient two-slot quartets for {speaker}')
        source_audits[speaker] = audit
        speaker_counts[speaker] = {
            'audio_alignment_video_codes': len(audio),
            'candidate_quartets': len(rows),
        }
        missing_anchors[speaker] = missing
        all_rows.extend(rows)
    return select_pilot(all_rows), {
        'speakers': list(SPEAKERS),
        'quartets_per_speaker': PER_SPEAKER,
        'speaker_counts': speaker_counts,
        'missing_neighbor_anchors': missing_anchors,
        'source_audits': source_audits,
        'media_content_decoded_or_semantically_inspected': False,
        'audio_archive_payload_bytes_hashed': True,
        'video_payload_bytes_read_by_audit': 0,
        'predictions_read': False,
        'sealed_audit_read': False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-root', type=Path, required=True)
    parser.add_argument('--frozen-validation', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() or output.is_relative_to(REPO):
        raise ValueError('use a new external output directory')
    selected, summary = build_from_bundles(args.bundle_root, args.frozen_validation)
    output.mkdir(parents=True)
    selected_path = output / 'selected.jsonl'
    with selected_path.open('x', encoding='utf-8', newline='\n') as stream:
        for row in selected:
            stream.write(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n')
    summary.update({
        'scope': 'corrected pre-media development pilot from direct per-talker bundles',
        'selected_quartets': len(selected),
        'selected_query_roles': sum(len(row['queries']) for row in selected),
        'selection_sha256': sha256_path(selected_path),
        'frozen_validation_sha256': sha256_path(args.frozen_validation),
        'script_sha256': sha256_path(Path(__file__)),
    })
    (output / 'summary.json').write_text(
        json.dumps(summary, indent=2) + '\n', encoding='utf-8',
    )
    print(json.dumps({
        'output_dir': str(output),
        'selected_quartets': summary['selected_quartets'],
        'selected_query_roles': summary['selected_query_roles'],
        'selection_sha256': summary['selection_sha256'],
    }))


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


if __name__ == '__main__':
    main()
