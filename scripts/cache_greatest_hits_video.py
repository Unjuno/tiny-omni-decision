"""Complete frozen native-task video banks; fresh output, pinned smoke, no learning."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import UTC, datetime
from itertools import groupby
from pathlib import Path

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import psutil

from scripts.greatest_hits_cache import frame_windows, frozen_events
from scripts.greatest_hits_full_extent import prepare_review_video
from scripts.greatest_hits_inputs import crop_audio
from scripts.native_full_extent import encoder_distribution_versions
from scripts.native_video_cache import sha
from scripts.probe_greatest_hits_encoders import check_source
from tiny_omni_decision.video_token_grid import token_grid

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('C:/CodexArtifacts/pretrained-reuse-r2')
TASKS = ROOT / 'greatest-hits-development-tasks-01'
MEDIA = ROOT / 'greatest-hits-development-media-01/members'
PROBE = ROOT / 'greatest-hits-full-extent-01'
PROBE_SHA = '1ad1b239fd8ffcf561f287e63b88f50c4e66025a437e2e8bcd4b19b7fb3a5f2f'
REVIEW = ROOT / 'greatest-hits-source-review-01'
REVIEW_SHA = '7f68821211c364cb84eaa24d82e41ca3a5a25d9cafb6a30e9e365dcbdfcee2be'
REVISION = 'c6eba0ece3dd49a24ffc44582966997a4a59f1d8'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('smoke', 'full'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--reference-sha256')
    args = parser.parse_args()
    if (args.phase == 'full') != bool(args.reference) or bool(args.reference) != bool(
            args.reference_sha256):
        raise ValueError('full mode requires successful same-source smoke and receipt SHA')
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO).strip()):
        raise ValueError('clean committed source and CPU-only process required')
    args.output.mkdir(exist_ok=False)
    tick = time.perf_counter()
    result = dict(status='RUNNING', phase=args.phase, pid=os.getpid(),
        command=[sys.executable, *sys.argv], source_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        started_utc=datetime.now(UTC).isoformat(), completed_events=0, completed_parents=0,
        completed_questions=0, matched_reference_events=0, optimizer_updates=0,
        device='cpu', dtype='float32', validation_loaded=args.phase == 'full',
        sealed_audit_loaded=False, calibration_loaded=False, quality_evaluated=False,
        cloud_cost=0, artifacts={}, parents=[])
    pins = {}

    def pin(path, expected=None):
        path = Path(path)
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError('pin mismatch: '+str(path))
        pins[str(path)] = actual

    def guard():
        result['peak_working_set_bytes'] = psutil.Process().memory_info().peak_wset
        if (time.perf_counter()-tick > (900 if args.phase == 'smoke' else 28800)
                or result['peak_working_set_bytes'] > 8*2**30
                or shutil.disk_usage(args.output).free < 16*2**30):
            raise RuntimeError('fixed wall/memory/disk resource guard')

    try:
        guard()
        pin(PROBE/'result.json', PROBE_SHA)
        pin(REVIEW/'result.json', REVIEW_SHA)
        pin(TASKS/'result.json',
            '7ef21536a84904df946f96cd774df9694b99027b1f36d6c8221b8b164fe7a831')
        pin(ROOT/'greatest-hits-development-decode-01/result.json',
            'fa18ca0958846c6143007c252960239a25e0735267f1a35f48d563ed4cfc7e76')
        task_receipt, probe, review = read(TASKS/'result.json'), read(PROBE/'result.json'), read(
            REVIEW/'result.json')
        if (task_receipt['status'] != 'PASS_FIXED_DEVELOPMENT_TASKS'
                or probe['status'] != 'PASS_FIXED_FULL_EXTENT_ENCODER'):
            raise ValueError('complete pinned tasks and full-extent probe required')
        splits = {}
        for split in ('train', 'validation'):
            path = TASKS/(split+'.jsonl')
            pin(path, task_receipt['artifacts'][path.name])
            splits[split] = [json.loads(line) for line in path.read_text().splitlines()]
        events = frozen_events(splits)
        question_counts = {s: len(rows) for s, rows in splits.items()}
        if (dict(Counter(x['split'] for x in events)) != dict(train=8102, validation=3049)
                or question_counts != dict(train=23707, validation=8875)):
            raise ValueError('complete frozen corpus counts required')
        references = {(r['parent'], r['event_row']+1): r for r in review['rows']}
        if args.phase == 'smoke':
            events = [x for x in events if (x['parent'], x['source_line']) in references]
            if len(events) != 21 or any(x['split'] != 'train' for x in events):
                raise ValueError('fixed 21 TRAIN smoke events required')
        for directory, receipt in [(PROBE, probe), (REVIEW, review)]:
            for name, digest in receipt['artifacts'].items():
                pin(directory/name, digest)
        for event in {x['parent']: x for x in events}.values():
            pin(MEDIA/event['media_file'], event['asset_sha256'])
        for name in subprocess.check_output(['git', 'ls-files', '-z', '--', '*.py'], cwd=REPO,
                                            text=True).split('\0'):
            if name:
                pin(REPO/name)
        source = ROOT.parent/'pretrained-reuse-r1/videoprism-base-r1/source'
        weights = ROOT.parent/'pretrained-reuse-r1/videoprism-base-f16r288.npz'
        pin(weights, '61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29')
        check_source(source, REVISION, 'videoprism')
        for path in source.rglob('*.py'):
            pin(path)
        estimated = len(events)*((2048*768+2048*3)*4+8*8)
        if shutil.disk_usage(args.output).free < estimated+16*2**30:
            raise RuntimeError('insufficient space for complete cache plus reserve')
        write(args.output/'plan.json', events)
        result.update(pins=pins, expected_events=len(events),
            expected_parents=len({x['parent'] for x in events}),
            expected_questions=sum(x['questions'] for x in events),
            estimated_tensor_bytes=estimated,
            plan_sha256=sha(args.output/'plan.json'), probe_receipt_sha256=PROBE_SHA,
            source_revision=REVISION, reference_sha256=args.reference_sha256)
        reference = None
        if args.reference:
            pin(args.reference/'result.json', args.reference_sha256)
            reference = read(args.reference/'result.json')
            if (reference['status'] != 'PASS_COMPLETE_VIDEO_CACHE' or reference['phase'] != 'smoke'
                    or reference['source_commit'] != result['source_commit']
                    or reference['matched_reference_events'] != 21):
                raise ValueError('successful exact same-source smoke required')
            for path, digest in reference['pins'].items():
                pin(path, digest)
            for name, digest in reference['artifacts'].items():
                pin(args.reference/name, digest)
        sys.path.insert(0, str(source))
        import av
        import jax
        import jax.numpy as jnp
        import tensorflow as tf
        from videoprism import models as vp

        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        if any(d.platform != 'cpu' for d in jax.devices()):
            raise ValueError('CPU-only encoder required')
        model = vp.get_model('videoprism_public_v1_base')
        variables = vp.load_pretrained_weights('videoprism_public_v1_base',
                                               checkpoint_path=str(weights))
        result['parameters'] = sum(int(x.size) for x in jax.tree_util.tree_leaves(variables))
        if result['parameters'] != 114365184 or any(
                np.asarray(x).dtype != np.float32 for x in jax.tree_util.tree_leaves(variables)):
            raise ValueError('fixed FP32 encoder inventory required')

        def state_digest():
            h = hashlib.sha256()
            for value in jax.tree_util.tree_leaves(variables):
                x = np.asarray(value)
                h.update(str(x.shape).encode())
                h.update(x.tobytes())
            return h.hexdigest()

        result['frozen_before'] = state_digest()
        result['versions'] = encoder_distribution_versions()
        if (result['frozen_before'] != probe['frozen_before']
                or result['versions'] != probe['versions']
                or reference and reference['versions'] != result['versions']):
            raise ValueError('encoder state/runtime differs from passing probe')
        encode = jax.jit(lambda x: model.apply(variables, x, train=False))
        result['load_seconds'] = time.perf_counter()-tick
        write(args.output/'result.json', result)
        for parent, grouped in groupby(events, key=lambda x: x['parent']):
            group = list(grouped)
            path = MEDIA/group[0]['media_file']
            guard()
            parent_tick = time.perf_counter()
            chunks, first, previous_end = [], None, None
            with av.open(str(path)) as container:
                container.streams.audio[0].codec_context.thread_count = 1
                for frame in container.decode(audio=0):
                    stamp = float(frame.pts*frame.time_base)
                    if frame.sample_rate != 96000 or len(frame.layout.channels) != 1:
                        raise ValueError('native 96k mono required')
                    if previous_end is not None and abs(stamp-previous_end) > 1e-8:
                        raise ValueError('discontinuous native audio')
                    first = stamp if first is None else first
                    previous_end = stamp+frame.samples/frame.sample_rate
                    chunks.append(frame.to_ndarray().reshape(-1))
            pcm = np.concatenate(chunks)
            timing = [crop_audio(pcm, 96000, first, x['event_seconds'])[1] for x in group]
            windows = [(x['start_seconds'], x['end_seconds']) for x in timing]
            bank_path = args.output/(parent+'-video.npy')
            if bank_path.exists():
                raise ValueError('output bank already exists')
            bank = np.lib.format.open_memmap(bank_path, mode='w+', dtype=np.float32,
                                             shape=(len(group), 2048, 768))
            positions, stamps, records, token_digests = [], [], [], []
            with av.open(str(path)) as container:
                container.streams.video[0].codec_context.thread_count = 1
                frames = ((float(f.pts*f.time_base), f.to_ndarray(format='rgb24'))
                          for f in container.decode(video=0))
                for i, ((rgb, pts), event, clock) in enumerate(zip(
                        frame_windows(frames, windows), group, timing, strict=True)):
                    guard()
                    prepared, pos, metadata = prepare_review_video(rgb, pts,
                        clock['start_seconds'], clock['end_seconds'], tf)
                    output = encode(jnp.asarray(prepared[None]))
                    jax.block_until_ready(output)
                    value = output[0] if isinstance(output, tuple) else output
                    tokens = token_grid(np.asarray(value)).reshape(2048, 768)
                    if tokens.dtype != np.float32 or not np.isfinite(tokens).all():
                        raise ValueError('finite FP32 features required')
                    bank[i] = tokens
                    np.testing.assert_array_equal(bank[i], tokens)
                    token_digests.append(hashlib.sha256(tokens.tobytes()).hexdigest())
                    positions.append(pos)
                    stamps.append(pts[metadata['selected_indices']])
                    key = (parent, event['source_line'])
                    if key in references:
                        ref = references[key]
                        prefix = f"{ref['number']:02d}-{parent}"
                        for suffix, arr in [('encoder-input', prepared), ('video', tokens),
                                            ('positions', pos), ('pts', stamps[-1])]:
                            np.testing.assert_array_equal(arr, np.load(
                                PROBE/(prefix+'-'+suffix+'.npy'), allow_pickle=False))
                        result['matched_reference_events'] += 1
                    records.append(dict(**event, bank_index=i, timing=clock,
                                        selected_pts=metadata['selected_pts']))
                    result['completed_events'] += 1
                    result['completed_questions'] += event['questions']
            bank.flush()
            del bank
            reloaded = np.load(bank_path, mmap_mode='r', allow_pickle=False)
            for i, digest in enumerate(token_digests):
                if hashlib.sha256(reloaded[i].tobytes()).hexdigest() != digest:
                    raise ValueError('flushed bank differs from encoder features')
            del reloaded
            for suffix, arrays in [('positions', positions), ('pts', stamps)]:
                dest = args.output/(parent+'-'+suffix+'.npy')
                with dest.open('xb') as stream:
                    np.save(stream, np.asarray(arrays), allow_pickle=False)
                np.testing.assert_array_equal(np.asarray(arrays), np.load(dest, allow_pickle=False))
            for suffix in ('video', 'positions', 'pts'):
                dest = args.output/(parent+'-'+suffix+'.npy')
                result['artifacts'][dest.name] = sha(dest)
            if sha(path) != group[0]['asset_sha256']:
                raise ValueError('media changed during parent extraction')
            result['parents'].append(dict(parent=parent, split=group[0]['split'], events=records,
                                           seconds=time.perf_counter()-parent_tick))
            result['completed_parents'] += 1
            result['elapsed_seconds'] = time.perf_counter()-tick
            write(args.output/'result.json', result)
            print(result['completed_parents'], '/', result['expected_parents'],
                  result['completed_events'], '/', len(events), parent, flush=True)
        result['frozen_after'] = state_digest()
        if (result['frozen_before'] != result['frozen_after']
                or result['completed_events'] != len(events)
                or result['completed_questions'] != result['expected_questions']
                or result['completed_parents'] != result['expected_parents']
                or result['matched_reference_events'] != 21
                or sha(args.output/'plan.json') != result['plan_sha256']):
            raise ValueError('incomplete coverage or changed frozen encoder')
        check_source(source, REVISION, 'videoprism')
        for path, digest in pins.items():
            if sha(Path(path)) != digest:
                raise ValueError('source/input changed while caching')
        guard()
        result['status'] = 'PASS_COMPLETE_VIDEO_CACHE'
    except BaseException as exc:
        result.update(status='INCOMPLETE', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        result.update(seconds=time.perf_counter()-tick, finished_utc=datetime.now(UTC).isoformat())
        write(args.output/'result.json', result)
        print(result['status'], flush=True)


if __name__ == '__main__':
    main()
