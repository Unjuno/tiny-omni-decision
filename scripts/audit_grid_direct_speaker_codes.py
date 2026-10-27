"""Compare GRID per-speaker source filenames without reading media payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import tarfile
from pathlib import Path, PurePosixPath

CODE_RE = re.compile(r'^[a-z0-9]{6}$')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tar_codes(path: Path, suffix: str) -> set[str]:
    """Read archive headers only and return six-character utterance IDs."""
    result = set()
    with tarfile.open(path, 'r:*') as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            if not member.isfile() or name.name.startswith('._'):
                continue
            if name.suffix.lower() != suffix:
                continue
            code = name.stem
            if not CODE_RE.fullmatch(code):
                raise ValueError(f'unexpected GRID utterance filename: {member.name}')
            if code in result:
                raise ValueError(f'duplicate GRID utterance code: {code}')
            result.add(code)
    return result


def zip_codes_from_tail(path: Path, *, range_start: int, total_size: int) -> tuple[set[str], int]:
    """Parse ZIP central-directory names from a tail range ending at EOF."""
    data = path.read_bytes()
    if range_start < 0 or range_start + len(data) != total_size:
        raise ValueError('ZIP metadata range must be contiguous through end of file')
    eocd = data.rfind(b'PK\x05\x06')
    if eocd < 0 or eocd + 22 > len(data):
        raise ValueError('ZIP end-of-central-directory record is missing')
    (_, disk, cd_disk, disk_count, total_count, cd_size, cd_offset,
     comment_length) = struct.unpack_from('<4s4H2LH', data, eocd)
    if eocd + 22 + comment_length != len(data):
        raise ValueError('ZIP comment or end-of-file boundary is inconsistent')
    if disk or cd_disk or disk_count != total_count or 0xFFFF in (disk_count, total_count):
        raise ValueError('multi-disk and ZIP64 catalogs are unsupported')
    if cd_offset < range_start or cd_offset + cd_size > total_size:
        raise ValueError('ZIP central directory is not fully present in the supplied tail')
    position = cd_offset - range_start
    end = position + cd_size
    codes = set()
    entries = 0
    while position < end:
        if data[position:position + 4] != b'PK\x01\x02' or position + 46 > end:
            raise ValueError('invalid ZIP central-directory header')
        fields = struct.unpack_from('<4s6H3L5H2L', data, position)
        name_length, extra_length, entry_comment_length = fields[10:13]
        record_end = position + 46 + name_length + extra_length + entry_comment_length
        if record_end > end:
            raise ValueError('truncated ZIP central-directory entry')
        flags = fields[3]
        encoding = 'utf-8' if flags & 0x800 else 'cp437'
        name = data[position + 46:position + 46 + name_length].decode(encoding)
        position = record_end
        entries += 1
        path_name = PurePosixPath(name)
        if path_name.name.startswith('._') or path_name.suffix.lower() != '.mpg':
            continue
        code = path_name.stem
        if not CODE_RE.fullmatch(code):
            raise ValueError(f'unexpected GRID video filename: {name}')
        if code in codes:
            raise ValueError(f'duplicate GRID video code: {code}')
        codes.add(code)
    if position != end or entries != total_count:
        raise ValueError('ZIP central-directory entry count or size is inconsistent')
    return codes, entries


def audit_speaker_bundle(
    *,
    audio_tar: Path,
    alignment_tar: Path,
    video_tail: Path,
    video_range_start: int,
    video_total_size: int,
    frozen_validation: Path,
    speaker: str,
) -> dict:
    audio = tar_codes(audio_tar, '.wav')
    alignments = tar_codes(alignment_tar, '.align')
    video, video_entries = zip_codes_from_tail(
        video_tail, range_start=video_range_start, total_size=video_total_size,
    )
    rows = json.loads(frozen_validation.read_bytes())
    frozen = {
        row['code'] for row in rows
        if isinstance(row, dict) and row.get('speaker') == speaker
    }
    if not frozen or any(not CODE_RE.fullmatch(code) for code in frozen):
        raise ValueError('frozen validation manifest has no valid rows for speaker')
    all_sets = {'audio': audio, 'alignment': alignments, 'video': video}
    overlaps = {
        'audio_alignment': len(audio & alignments),
        'audio_video': len(audio & video),
        'alignment_video': len(alignments & video),
    }
    source_match = len(audio) == len(alignments) == len(video) and all(
        len(audio & codes) == len(audio) for name, codes in all_sets.items()
        if name != 'audio'
    )
    frozen_match = all(frozen <= codes for codes in all_sets.values())
    if not source_match:
        status = 'BLOCKED_DIRECT_SOURCE_BUNDLE_MISMATCH'
    elif not frozen_match:
        status = 'OFFICIAL_BUNDLE_MATCH_FROZEN_SPLIT_MISMATCH'
    else:
        status = 'PASS_DIRECT_SOURCE_BUNDLE_AND_FROZEN_SPLIT'
    return {
        'status': status,
        'speaker': speaker,
        'source_counts': {
            'audio_wav_codes': len(audio),
            'alignment_codes': len(alignments),
            'video_mpg_codes': len(video),
            'video_zip_entries_including_sidecars': video_entries,
        },
        'source_filename_overlaps': overlaps,
        'frozen_validation_rows': len(frozen),
        'frozen_validation_overlap': {
            name: len(frozen & codes) for name, codes in all_sets.items()
        },
        'input_sha256': {
            'audio_tar': sha256_file(audio_tar),
            'alignment_tar': sha256_file(alignment_tar),
            'video_metadata_tail': sha256_file(video_tail),
            'frozen_validation': sha256_file(frozen_validation),
        },
        'video_zip': {
            'total_size_bytes': video_total_size,
            'metadata_range_start': video_range_start,
            'metadata_range_bytes': video_tail.stat().st_size,
        },
        'media_content_decoded_or_semantically_inspected': False,
        'audio_archive_payload_bytes_hashed': True,
        'video_payload_bytes_read_by_audit': 0,
        'speaker_identity_inferred': False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio-tar', type=Path, required=True)
    parser.add_argument('--alignment-tar', type=Path, required=True)
    parser.add_argument('--video-metadata-tail', type=Path, required=True)
    parser.add_argument('--video-range-start', type=int, required=True)
    parser.add_argument('--video-total-size', type=int, required=True)
    parser.add_argument('--frozen-validation', type=Path, required=True)
    parser.add_argument('--speaker', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit_speaker_bundle(
        audio_tar=args.audio_tar,
        alignment_tar=args.alignment_tar,
        video_tail=args.video_metadata_tail,
        video_range_start=args.video_range_start,
        video_total_size=args.video_total_size,
        frozen_validation=args.frozen_validation,
        speaker=args.speaker,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'output': str(args.output)}))
    if result['status'] != 'PASS_DIRECT_SOURCE_BUNDLE_AND_FROZEN_SPLIT':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
