"""Complete existing development audio identities; no held-out file discovery."""

import hashlib
import io
import zipfile
from collections import defaultdict
from contextlib import closing
from pathlib import Path

from scripts.lombard_grid_corpus import decode_code
from scripts.starss23_stream import audio_windows, validate_intervals

SPLITS = ('train', 'validation')


def _bind(owners, split, kind, value):
    previous = owners.setdefault((kind, value), split)
    if previous != split:
        raise ValueError('cross-split ' + kind + ' overlap')


def bind_pcm(owners, split, rate, digest):
    _bind(owners, split, 'decoded_mono_pcm', (rate, digest))


def audio_plan(lombard, media, starss):
    if set(lombard) != set(SPLITS) or set(starss) != set(SPLITS):
        raise ValueError('exactly train/validation source files required')
    media_by_id = {r['source_id']: r for r in media}
    if len(media_by_id) != len(media):
        raise ValueError('duplicate media metadata source IDs')
    owners, seen, rows, shapes, counts = {}, set(), [], {}, {}
    for split in SPLITS:
        if not lombard[split] or not starss[split]:
            raise ValueError('both existing sources required in each split')
        current = []
        for row in lombard[split]:
            item = media_by_id[row['source_id']]
            if (row['split'] != split or item['split'] != split
                    or row['label_status'] != 'eligible'
                    or row['targets'] != decode_code(row['actual_code'])
                    or item['audio_status'] != 'PASS' or item['video_status'] != 'PASS'
                    or item['rate'] != 16000 or not 0 < item['samples'] <= 480000):
                raise ValueError('Lombard metadata/labels differ from admitted profile')
            for kind in ('source_id', 'speaker', 'speaker_group'):
                _bind(owners, split, 'lombard_' + kind, row[kind])
            _bind(owners, split, 'lombard_phrase', row['actual_code'])
            try:
                decode_code(row['intended_code'])
            except ValueError:
                pass  # Retain the historical handling of unresolved intended annotations.
            else:
                _bind(owners, split, 'lombard_phrase', row['intended_code'])
            _bind(owners, split, 'media_sha256', item['audio_sha256'])
            bind_pcm(owners, split, 16000, item['decoded_mono_f32_sha256'])
            current.append(dict(id='lombard:' + row['source_id'], source='lombard', split=split,
                source_id=row['source_id'], group=row['speaker_group'],
                member=row['assets']['audio']['name'], media_bytes=row['assets']['audio']['bytes'],
                media_sha256=item['audio_sha256'], pcm_sha256=item['decoded_mono_f32_sha256'],
                rate=16000, samples=item['samples']))
        windows, recording_media = defaultdict(list), {}
        for row in starss[split]:
            if (row['split'] != split or row['source_revision'] != 'doi:10.5281/zenodo.7880637'):
                raise ValueError('STARSS source revision or split differs')
            asset = row['media']['audio']
            for kind in ('room', 'recording'):
                _bind(owners, split, 'starss_' + kind, row[kind])
            _bind(owners, split, 'media_sha256', asset['sha256'])
            identity = (asset['path'], asset['sha256'], asset['bytes'])
            if recording_media.setdefault(row['recording'], identity) != identity:
                raise ValueError('one recording maps to different physical media')
            windows[row['recording']].append((row['start_seconds'], row['end_seconds']))
            current.append(dict(id='starss:' + row['id'], source='starss', split=split,
                source_id=row['recording'], group=row['room'], path=asset['path'],
                media_sha256=asset['sha256'], media_bytes=asset['bytes'],
                start_seconds=row['start_seconds'], end_seconds=row['end_seconds'],
                rate=24000, samples=120000))
        for intervals in windows.values():
            validate_intervals(intervals)
        speech_offset, event_offset = 0, 0
        for row in current:
            if row['id'] in seen:
                raise ValueError('duplicate audio example ID')
            seen.add(row['id'])
            count = (row['samples'] * 16000 + row['rate'] * 320 - 1) // (row['rate'] * 320)
            row.update(speech_tokens=count, speech_offset=speech_offset, event_offset=event_offset)
            speech_offset += count
            event_offset += 1
        rows.extend(current)
        shapes[split + '-speech.npy'] = [speech_offset, 384]
        shapes[split + '-event.npy'] = [event_offset, 384]
        counts[split] = {}
        for source in ('lombard', 'starss'):
            selected = [r for r in current if r['source'] == source]
            counts[split][source] = dict(examples=len(selected),
                assets=len({r['media_sha256'] for r in selected}),
                groups=len({r['group'] for r in selected}),
                speech_tokens=sum(r['speech_tokens'] for r in selected),
                physical_seconds=sum(r['samples'] / r['rate'] for r in selected))
    return dict(rows=rows, shapes=shapes, counts=counts,
                tensor_bytes=sum(n * d * 4 for n, d in shapes.values()))


def iter_audio(rows, archive_path):
    """Decode only explicit admitted rows; do not extract an archive to disk."""
    import numpy as np
    import soundfile as sf

    owners, physical = {}, {}

    def verify_file(path, expected):
        with path.open('rb') as handle:
            actual = (path.stat().st_size, hashlib.file_digest(handle, 'sha256').hexdigest())
        if actual != expected:
            raise ValueError('STARSS physical audio bytes/hash mismatch')

    with zipfile.ZipFile(archive_path) as archive:
        for row in rows:
            if row['source'] == 'lombard':
                content = archive.read(row['member'])
                if (len(content) != row['media_bytes']
                        or hashlib.sha256(content).hexdigest() != row['media_sha256']):
                    raise ValueError('audio archive member bytes/hash mismatch')
                waveform, rate = sf.read(io.BytesIO(content), dtype='float32')
            elif row['source'] == 'starss':
                path = Path(row['path'])
                expected = (row['media_bytes'], row['media_sha256'])
                if path not in physical:
                    verify_file(path, expected)
                    physical[path] = expected
                elif physical[path] != expected:
                    raise ValueError('one physical audio file has conflicting identities')
                with closing(audio_windows(path,
                             [(row['start_seconds'], row['end_seconds'])])) as windows:
                    frames = next(windows)
                waveform, rate = frames[:, 0].copy(), 24000
            else:
                raise ValueError('unknown admitted audio source')
            if (waveform.shape != (row['samples'],) or rate != row['rate']
                    or waveform.dtype != np.float32 or not np.isfinite(waveform).all()):
                raise ValueError('decoded waveform shape/rate/dtype/finiteness differs')
            digest = hashlib.sha256(waveform.tobytes()).hexdigest()
            if row.get('pcm_sha256', digest) != digest:
                raise ValueError('historical decoded PCM differs')
            bind_pcm(owners, row['split'], rate, digest)
            yield row, waveform, digest
    for path, expected in physical.items():
        verify_file(path, expected)
