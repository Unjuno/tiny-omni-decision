"""Decode a prospectively fixed TRAIN parent per date for qualitative source review."""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from scripts.greatest_hits_inputs import crop_audio
from scripts.greatest_hits_review import acoustic_evidence, review_event, review_parents
from scripts.greatest_hits_source import parse_times
from scripts.native_video_cache import sha


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo).strip()):
        raise ValueError('clean committed source and CPU-only process required')
    import av
    import numpy as np
    import psutil

    root = Path('C:/CodexArtifacts/pretrained-reuse-r2')
    inventory = root / 'greatest-hits-train-inventory-01'
    media = root / 'greatest-hits-development-media-01'
    audit = root / 'greatest-hits-development-decode-01'
    pins = {}

    def pin(path, expected=None):
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError('source pin mismatch: ' + str(path))
        pins[str(path)] = actual

    pin(inventory / 'partition.json',
        '6c9ac8eb8e12bd5dc0af77c8d162a40c753d5a215d07eb85dc3a69c3973d9f98')
    pin(audit / 'result.json',
        'fa18ca0958846c6143007c252960239a25e0735267f1a35f48d563ed4cfc7e76')
    if read(audit / 'result.json')['status'] != 'PASS_COMPLETE_DECODE':
        raise ValueError('complete fixed-media integrity required')
    partition = read(inventory / 'partition.json')
    selected = review_parents(partition['train'])
    if len(selected) != 21 or set(selected) & set(partition['validation']):
        raise ValueError('exact21 TRAIN-date cohort required')
    members, media_members = read(inventory / 'members.json'), read(media / 'members.json')
    pin(inventory / 'members.json')
    pin(media / 'members.json')
    plan = []
    for ident in selected:
        path = inventory / 'members' / (ident + '_times.txt')
        pin(path, members[ident]['sha256'])
        rows = parse_times(path.read_bytes())
        index = review_event(rows)
        if media_members[ident]['split'] != 'train':
            raise ValueError('selected media is not TRAIN')
        pin(media / 'members' / media_members[ident]['file'], media_members[ident]['sha256'])
        plan.append(dict(parent=ident, event_row=index, **rows[index]))
    for name in ('scripts/inspect_greatest_hits_source_review.py',
                 'scripts/greatest_hits_review.py', 'scripts/greatest_hits_inputs.py',
                 'scripts/greatest_hits_source.py'):
        pin(repo / name)
    args.output.mkdir(exist_ok=False)
    write(args.output / 'plan.json', plan)  # Fixed and persisted before any media decoding.
    started = time.perf_counter()
    result = dict(status='RUNNING', command=[sys.executable, *sys.argv],
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo,
                                             text=True).strip(),
        started_utc=datetime.now(UTC).isoformat(), python=platform.python_version(),
        pyav=av.__version__, numpy=np.__version__, pins=pins,
        plan_sha256=sha(args.output/'plan.json'),
        rows=[], artifacts={}, models_loaded=0, optimizer_updates=0, quality_evaluation=False,
        validation_media_loaded=False, validation_annotation_members_loaded=False,
        test_media_or_annotations_loaded=False, sealed_audit_loaded=False)

    def save(name, values):
        path = args.output / (name + '.npy')
        with path.open('xb') as stream:
            np.save(stream, values, allow_pickle=False)
        np.testing.assert_array_equal(values, np.load(path, allow_pickle=False))
        result['artifacts'][path.name] = sha(path)

    try:
        for number, row in enumerate(plan):
            if (time.perf_counter()-started > 600 or shutil.disk_usage(args.output).free < 2*2**30
                    or psutil.Process().memory_info().peak_wset > 8*2**30):
                raise RuntimeError('fixed600s/2GiB free/8GiB host resource guard')
            ident = row['parent']
            path = media / 'members' / media_members[ident]['file']
            chunks, first, previous_end = [], None, None
            with av.open(str(path)) as container:
                container.streams.audio[0].codec_context.thread_count = 1
                for frame in container.decode(audio=0):
                    stamp = float(frame.pts * frame.time_base)
                    if frame.sample_rate != 96000 or len(frame.layout.channels) != 1:
                        raise ValueError('native96k mono required')
                    if previous_end is not None and abs(stamp-previous_end) > 1e-8:
                        raise ValueError('discontinuous native audio')
                    first = stamp if first is None else first
                    previous_end = stamp + frame.samples/frame.sample_rate
                    chunks.append(frame.to_ndarray().reshape(-1))
            pcm, timing = crop_audio(np.concatenate(chunks), 96000, first, row['seconds'])
            frames, pts = [], []
            with av.open(str(path)) as container:
                container.streams.video[0].codec_context.thread_count = 1
                for frame in container.decode(video=0):
                    stamp = float(frame.pts * frame.time_base)
                    if timing['start_seconds'] <= stamp < timing['end_seconds']:
                        frames.append(frame.to_ndarray(format='rgb24'))
                        pts.append(stamp)
                    if stamp >= timing['end_seconds']:
                        break
            rgb = np.stack(frames)
            if (not 8 <= len(frames) <= 16 or rgb.shape[1:] != (256, 456, 3)
                    or not (np.diff(pts) > 0).all()):
                raise ValueError('ordered native event frames required')
            name = f'{number:02d}-{ident}'
            save(name+'-pcm96', pcm)
            save(name+'-rgb', rgb)
            save(name+'-pts', np.asarray(pts, np.float64))
            canvas = np.zeros((1024, 1824, 3), np.uint8)
            for i, frame in enumerate(rgb):
                y, x = (i//4)*256, (i%4)*456
                canvas[y:y+256, x:x+456] = frame
            preview = args.output / (name+'-contact.png')
            with av.open(str(preview), mode='w', format='image2') as output:
                stream = output.add_stream('png', rate=1)
                stream.width, stream.height, stream.pix_fmt = 1824, 1024, 'rgb24'
                for packet in stream.encode(av.VideoFrame.from_ndarray(canvas, format='rgb24')):
                    output.mux(packet)
                for packet in stream.encode():
                    output.mux(packet)
            result['artifacts'][preview.name] = sha(preview)
            result['rows'].append(dict(**row, number=number, timing=timing, video_pts=pts,
                preview=preview.name, acoustic=acoustic_evidence(pcm, timing['start_seconds'],
                                                               row['seconds'])))
            print(number+1, '/21', ident, flush=True)
            write(args.output/'result.json', result)
        for path, digest in pins.items():
            if sha(Path(path)) != digest:
                raise ValueError('source changed during inspection')
        result['status'] = 'PASS_FIXED_TRAIN_SOURCE_EXTRACTION'
    except BaseException as exc:
        result.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        result.update(seconds=time.perf_counter()-started,
                      finished_utc=datetime.now(UTC).isoformat(),
                      peak_working_set_bytes=psutil.Process().memory_info().peak_wset)
        write(args.output/'result.json', result)
        print(result['status'], flush=True)


if __name__ == '__main__':
    main()
