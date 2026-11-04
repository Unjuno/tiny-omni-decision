"""Execute frozen full-extent VideoPrism on the fixed 21 TRAIN review events."""

import argparse
import hashlib
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

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import psutil

from scripts.greatest_hits_full_extent import prepare_review_video
from scripts.native_full_extent import encoder_distribution_versions
from scripts.native_video_cache import sha
from scripts.probe_greatest_hits_encoders import check_source
from tiny_omni_decision.video_token_grid import token_grid

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('C:/CodexArtifacts/pretrained-reuse-r2')
INPUT = ROOT / 'greatest-hits-source-review-01'
INPUT_SHA = '7f68821211c364cb84eaa24d82e41ca3a5a25d9cafb6a30e9e365dcbdfcee2be'
PLAN_SHA = '951d1078d3fae228256bb398bd636fea2393fd7aa4b05969ef04655c7f475ae5'
REVISION = 'c6eba0ece3dd49a24ffc44582966997a4a59f1d8'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--reference-sha256')
    args = parser.parse_args()
    if bool(args.reference) != bool(args.reference_sha256):
        raise ValueError('reference directory and receipt SHA must be paired')
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO).strip()):
        raise ValueError('clean committed source and isolated CPU process required')
    args.output.mkdir(exist_ok=False)
    started = time.perf_counter()
    result = dict(status='RUNNING', command=[sys.executable, *sys.argv], pid=os.getpid(),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                                             text=True).strip(),
        started_utc=datetime.now(UTC).isoformat(), python=platform.python_version(),
        device='cpu', dtype='float32', external_source_revision=REVISION,
        optimizer_updates=0, quality_evaluation=False, validation_loaded=False,
        sealed_audit_loaded=False, calibration_loaded=False, cloud_cost=0, rows=[])
    pins, artifacts, reference = {}, {}, None

    def pin(path, expected=None):
        path = Path(path)
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError('pin mismatch: ' + str(path))
        pins[str(path)] = actual

    def guard():
        result['peak_working_set_bytes'] = psutil.Process().memory_info().peak_wset
        if (time.perf_counter() - started > 900 or shutil.disk_usage(args.output).free < 2*2**30
                or result['peak_working_set_bytes'] > 8*2**30):
            raise RuntimeError('fixed 900s / 8GiB host / 2GiB free resource guard')

    def record(path):
        artifacts[path.name] = sha(path)
        if reference and (artifacts[path.name] != reference['artifacts'].get(path.name)
                          or sha(args.reference / path.name) != artifacts[path.name]):
            raise ValueError('fresh-process replay differs: ' + path.name)

    def save(name, array):
        value = np.asarray(array)
        if value.dtype not in (np.dtype('float32'), np.dtype('float64')):
            raise ValueError('floating features/input/positions or original PTS required')
        if not np.isfinite(value).all():
            raise ValueError('nonfinite output')
        path = args.output / (name + '.npy')
        with path.open('xb') as stream:
            np.save(stream, value, allow_pickle=False)
        np.testing.assert_array_equal(value, np.load(path, allow_pickle=False))
        record(path)

    try:
        guard()
        pin(INPUT / 'result.json', INPUT_SHA)
        pin(INPUT / 'plan.json', PLAN_SHA)
        inputs, plan = read(INPUT/'result.json'), read(INPUT/'plan.json')
        if (inputs['status'] != 'PASS_FIXED_TRAIN_SOURCE_EXTRACTION'
                or inputs['plan_sha256'] != PLAN_SHA or len(plan) != 21
                or [r['parent'] for r in inputs['rows']] != [r['parent'] for r in plan]):
            raise ValueError('exact fixed 21 TRAIN review events required')
        # Original source pins are checked without decoding or relabeling the media.
        for path, expected in inputs['pins'].items():
            pin(path, expected)
        for name, expected in inputs['artifacts'].items():
            pin(INPUT / name, expected)
        for name in subprocess.check_output(['git', 'ls-files', '-z', '--', '*.py'], cwd=REPO,
                                            text=True).split('\0'):
            if name:
                pin(REPO / name)
        source = ROOT.parent / 'pretrained-reuse-r1/videoprism-base-r1/source'
        checkpoint = ROOT.parent / 'pretrained-reuse-r1/videoprism-base-f16r288.npz'
        check_source(source, REVISION, 'videoprism')
        pin(checkpoint, '61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29')
        for path in source.rglob('*.py'):
            pin(path)
        if args.reference:
            if sha(args.reference/'result.json') != args.reference_sha256:
                raise ValueError('reference receipt changed')
            reference = read(args.reference/'result.json')
            if reference['status'] != 'PASS_FIXED_FULL_EXTENT_ENCODER' or reference['pins'] != pins:
                raise ValueError('complete same-source/input reference required')
        result.update(pins=pins, input_receipt_sha256=INPUT_SHA,
                      reference_sha256=args.reference_sha256)
        sys.path.insert(0, str(source))
        import av
        import jax
        import jax.numpy as jnp
        import tensorflow as tf
        from videoprism import models as vp

        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        if any(d.platform != 'cpu' for d in jax.devices()):
            raise ValueError('CPU-only VideoPrism required')
        model = vp.get_model('videoprism_public_v1_base')
        variables = vp.load_pretrained_weights('videoprism_public_v1_base',
                                               checkpoint_path=str(checkpoint))
        result['parameters'] = sum(int(v.size) for v in jax.tree_util.tree_leaves(variables))
        if result['parameters'] != 114365184:
            raise ValueError('encoder parameter inventory changed')

        def digest():
            h = hashlib.sha256()
            for value in jax.tree_util.tree_leaves(variables):
                x = np.asarray(value)
                h.update(str(x.shape).encode())
                h.update(x.tobytes())
            return h.hexdigest()

        result['frozen_before'] = digest()
        result['versions'] = encoder_distribution_versions()
        encode = jax.jit(lambda x: model.apply(variables, x, train=False))
        result['load_seconds'] = time.perf_counter() - started
        write(args.output/'result.json', result)
        for row in inputs['rows']:
            guard()
            name = f"{row['number']:02d}-{row['parent']}"
            rgb = np.load(INPUT/(name+'-rgb.npy'), allow_pickle=False)
            pts = np.load(INPUT/(name+'-pts.npy'), allow_pickle=False)
            tick = time.perf_counter()
            prepared, positions, metadata = prepare_review_video(rgb, pts,
                row['timing']['start_seconds'], row['timing']['end_seconds'], tf)
            output = encode(jnp.asarray(prepared[None]))
            jax.block_until_ready(output)
            value = output[0] if isinstance(output, tuple) else output
            tokens = token_grid(np.asarray(value)).reshape(2048, 768)
            if tokens.dtype != np.float32:
                raise ValueError('FP32 features required')
            elapsed = time.perf_counter() - tick
            for suffix, data in [('encoder-input', prepared), ('video', tokens),
                                 ('positions', positions),
                                 ('pts', pts[metadata['selected_indices']])]:
                save(name+'-'+suffix, data)
            canvas = np.rint(prepared*255).astype(np.uint8).reshape(2, 4, 288, 288, 3)
            canvas = canvas.transpose(0, 2, 1, 3, 4).reshape(576, 1152, 3)
            preview = args.output/(name+'-prepared.png')
            with av.open(str(preview), mode='w', format='image2') as container:
                stream = container.add_stream('png', rate=1)
                stream.width, stream.height, stream.pix_fmt = 1152, 576, 'rgb24'
                for packet in stream.encode(av.VideoFrame.from_ndarray(canvas, format='rgb24')):
                    container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            record(preview)
            result['rows'].append(dict(parent=row['parent'], event_row=row['event_row'],
                number=row['number'], forward_seconds=elapsed, **metadata))
            print(row['number']+1, '/21', row['parent'], elapsed, flush=True)
            write(args.output/'result.json', result)
        result['frozen_after'] = digest()
        if result['frozen_before'] != result['frozen_after']:
            raise ValueError('frozen encoder state changed')
        if reference and (reference['versions'] != result['versions']
                or reference['frozen_before'] != result['frozen_before']
                or reference['artifacts'] != artifacts):
            raise ValueError('replay runtime, state or artifact inventory differs')
        check_source(source, REVISION, 'videoprism')
        for path, expected in pins.items():
            if sha(Path(path)) != expected:
                raise ValueError('source/input changed during execution')
        guard()
        result['status'] = 'PASS_FIXED_FULL_EXTENT_ENCODER'
    except BaseException as exc:
        result.update(status='INCOMPLETE', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        result.update(seconds=time.perf_counter()-started,
                      finished_utc=datetime.now(UTC).isoformat(), artifacts=artifacts)
        write(args.output/'result.json', result)
        print(result['status'], flush=True)


if __name__ == '__main__':
    main()
