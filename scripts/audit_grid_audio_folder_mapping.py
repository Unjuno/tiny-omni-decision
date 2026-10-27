"""Fail closed when official GRID alignment speakers do not map to audio folders."""

import argparse
import collections
import json
import re
import zipfile
from pathlib import Path, PurePosixPath

from scripts.audit_droid_train_temporal import digest

ROOT = Path('C:/CodexArtifacts/pretrained-reuse-r2')
SPLITS = ROOT / 'grid-development-split-20261028-attempt-02'
AUDIO = ROOT / 'grid-audio-20261028-attempt-01/audio_25k.zip'
SPLIT_PINS = {
    'train': 'f6f045931aee53e576fca22efda0bc888a983aed685a30edc38b2918ff156817',
    'validation': 'c1e77098327841fadc283bb14813c38de94e07fd16728d9cc3602bc261961e11',
}
AUDIO_SHA256 = 'd89331b1db49770ca0d0c676e75b6e35725ecb48cf35915e8482285af2594134'
SPEAKER_RE = re.compile(r'^s[0-9]+$')
CODE_RE = re.compile(r'^[a-z0-9]{6}$')
REPO = Path(__file__).resolve().parents[1]


def parse_audio_members(members):
    """Return speaker->codes from ZIP member names; never reads audio bytes."""
    audio = collections.defaultdict(set)
    for name in members:
        path = PurePosixPath(name)
        if (len(path.parts) != 3 or path.parts[0] != 'audio_25k'
                or path.suffix.lower() != '.wav' or path.name.startswith('._')):
            continue
        speaker, code = path.parts[1], path.stem
        if not SPEAKER_RE.fullmatch(speaker) or not CODE_RE.fullmatch(code):
            raise ValueError('unexpected GRID audio member name')
        if code in audio[speaker]:
            raise ValueError('duplicate audio code within speaker folder')
        audio[speaker].add(code)
    return dict(audio)


def audit_mapping(rows, audio_codes):
    """Compare alignment code identities to same-named audio folders without remapping."""
    aligned = collections.defaultdict(set)
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('alignment row must be an object')
        speaker, code = row.get('speaker'), row.get('code')
        if (not isinstance(speaker, str) or not SPEAKER_RE.fullmatch(speaker)
                or not isinstance(code, str) or not CODE_RE.fullmatch(code)):
            raise ValueError('invalid alignment speaker/code')
        identity = speaker, code
        if identity in seen:
            raise ValueError('duplicate alignment speaker/code row')
        seen.add(identity)
        aligned[speaker].add(code)
    if not aligned:
        raise ValueError('no alignment rows')
    if any(not SPEAKER_RE.fullmatch(speaker)
           or not isinstance(codes, set) for speaker, codes in audio_codes.items()):
        raise ValueError('invalid parsed audio-folder mapping')

    folder_matches = {}
    missing_total = 0
    for speaker in sorted(aligned):
        codes = aligned[speaker]
        own_codes = audio_codes.get(speaker, set())
        missing = sorted(codes - own_codes)
        missing_total += len(missing)
        matches = sorted(
            ((len(codes & folder_codes), folder)
             for folder, folder_codes in audio_codes.items()),
            key=lambda value: (-value[0], value[1]),
        )
        best_overlap, best_folder = matches[0] if matches else (0, None)
        folder_matches[speaker] = {
            'alignment_codes': len(codes),
            'same_name_folder_overlap': len(codes & own_codes),
            'same_name_folder_missing': len(missing),
            'best_audio_folder': best_folder,
            'best_overlap': best_overlap,
            'best_is_unique': len(matches) < 2 or matches[0][0] > matches[1][0],
        }
    status = ('PASS_AUDIO_FOLDER_MAPPING' if missing_total == 0
              else 'BLOCKED_AUDIO_SPEAKER_FOLDER_MAPPING')
    return {
        'status': status,
        'alignment_speakers': len(aligned),
        'alignment_codes': sum(map(len, aligned.values())),
        'audio_folders': len(audio_codes),
        'self_folder_missing_codes': missing_total,
        'folder_matches': folder_matches,
        'audio_folders_without_alignment_speaker': sorted(set(audio_codes) - set(aligned)),
        'media_bytes_read': False,
        'speaker_identity_inferred_or_remapped': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() or out.is_relative_to(REPO):
        raise ValueError('use a new external output directory')
    if any(digest(SPLITS / f'{part}.json') != expected
           for part, expected in SPLIT_PINS.items()):
        raise ValueError('pinned alignment split hash mismatch')
    if digest(AUDIO) != AUDIO_SHA256:
        raise ValueError('pinned official audio archive hash mismatch')
    rows = []
    split_speakers = {}
    for part in ('train', 'validation'):
        part_rows = json.loads((SPLITS / f'{part}.json').read_bytes())
        rows.extend(part_rows)
        split_speakers[part] = {row['speaker'] for row in part_rows}
    if split_speakers['train'] & split_speakers['validation']:
        raise ValueError('speaker split overlap')
    with zipfile.ZipFile(AUDIO) as archive:
        audio_codes = parse_audio_members(info.filename for info in archive.infolist())
    result = audit_mapping(rows, audio_codes)
    if any(digest(SPLITS / f'{part}.json') != expected
           for part, expected in SPLIT_PINS.items()) or digest(AUDIO) != AUDIO_SHA256:
        raise ValueError('pinned input changed during audit')
    result.update({
        'audio_archive_sha256': AUDIO_SHA256,
        'split_sha256': SPLIT_PINS,
        'split_speakers': {part: sorted(names) for part, names in split_speakers.items()},
        'script_sha256': digest(Path(__file__)),
    })
    out.mkdir(parents=True)
    (out / 'mapping-audit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'],
                      'self_folder_missing_codes': result['self_folder_missing_codes'],
                      'speakers': result['alignment_speakers'],
                      'output': str(out / 'mapping-audit.json')}, separators=(',', ':')))
    if result['status'] != 'PASS_AUDIO_FOLDER_MAPPING':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
