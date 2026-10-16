"""Three fixed TRAIN events through frozen audio or video encoders; no fit or scoring."""

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
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

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import psutil

from scripts.greatest_hits_features import event_positions, prepared_audio_tokens
from scripts.native_video_cache import sha

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('C:/CodexArtifacts/pretrained-reuse-r2')
R1 = ROOT.parent / 'pretrained-reuse-r1'
INPUT = ROOT / 'greatest-hits-input-physical-01'
INPUT_SHA = '97c91689effda1e7a080b81ddd4be7738718168493137a565f821371d493c0f6'
IDS = ['2015-03-30-00-26-27', '2015-03-12-20-15-29', '2015-03-31-02-08-23']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def check_source(source, revision, package):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()
    if git('rev-parse', 'HEAD') != revision or git('diff', 'HEAD', '--name-only'):
        raise ValueError('external model source differs')
    extra = git('ls-files', '--others', '--exclude-standard', '--', package).splitlines()
    if any(p.endswith(('.py', '.pyd', '.dll')) for p in extra):
        raise ValueError('untracked importable model source')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('audio', 'video'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--reference-sha256')
    args = parser.parse_args()
    if bool(args.reference) != bool(args.reference_sha256):
        raise ValueError('reference and receipt SHA must be paired')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO).strip():
        raise ValueError('clean committed source required')
    args.output.mkdir(exist_ok=False)
    started = time.perf_counter()
    result = dict(status='RUNNING', phase=args.phase, command=[sys.executable, *sys.argv],
        pid=os.getpid(), started_utc=datetime.now(UTC).isoformat(),
        python=platform.python_version(),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                                             text=True).strip(),
        optimizer_updates=0, quality_evaluation=False, validation_loaded=False,
        sealed_audit_loaded=False, calibration_loaded=False, cloud_cost=0, rows=[])
    pins, artifacts = {}, {}

    def pin(path, expected=None):
        path = Path(path)
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError('pin mismatch: ' + str(path))
        pins[str(path)] = actual

    def guard():
        result['os_peak_working_set_bytes'] = psutil.Process().memory_info().peak_wset
        if (time.perf_counter() - started > 900 or shutil.disk_usage(args.output).free < 2 * 2**30
                or result['os_peak_working_set_bytes'] > 8 * 2**30):
            raise RuntimeError('fixed900s/8GiB host/2GiB free resource guard')

    def save(name, array):
        value = np.asarray(array)
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('finite FP32 input/features/positions required')
        path = args.output / (name + '.npy')
        with path.open('xb') as stream:
            np.save(stream, value, allow_pickle=False)
        np.testing.assert_array_equal(value, np.load(path, allow_pickle=False))
        artifacts[path.name] = sha(path)
        if reference:
            old = args.reference / path.name
            if (artifacts[path.name] != reference['artifacts'].get(path.name)
                    or sha(old) != sha(path)):
                raise ValueError('fresh process replay differs: ' + path.name)

    try:
        guard()
        pin(INPUT / 'result.json', INPUT_SHA)
        inputs = read(INPUT / 'result.json')
        if (inputs['status'] != 'PASS_FIXED_INPUT_PROBE' or inputs['phase'] != 'physical'
                or [r['parent'] for r in inputs['rows']] != IDS
                or any(r['event_row'] != 0 for r in inputs['rows'])):
            raise ValueError('fixed three-TRAIN physical inputs required')
        for name, digest in inputs['artifacts'].items():
            pin(INPUT / name, digest)
        for name in subprocess.check_output(['git', 'ls-files', '-z', '--', '*.py'], cwd=REPO,
                                            text=True).split('\0'):
            if name:
                pin(REPO / name)
        audio_config_path = REPO / 'configs/pretrained_reuse/native_audio_profile_v1.json'
        audio_config = read(audio_config_path)
        pin(audio_config_path)
        if args.phase == 'audio':
            source = Path(audio_config['event_source'])
            revision, package = audio_config['event_revision'], 'models'
            speech_path = Path(audio_config['speech_model'])
            for path in [speech_path / n for n in ('config.json', 'preprocessor_config.json',
                                                  'encoder.safetensors')]:
                pin(path, audio_config['input_pins'][str(path)])
            pin(audio_config['event_weights'],
                audio_config['input_pins'][audio_config['event_weights']])
        else:
            source = R1 / 'videoprism-base-r1/source'
            revision, package = 'c6eba0ece3dd49a24ffc44582966997a4a59f1d8', 'videoprism'
            checkpoint = R1 / 'videoprism-base-f16r288.npz'
            pin(checkpoint, '61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29')
        check_source(source, revision, package)
        for p in source.rglob('*.py'):
            pin(p)
        reference = None
        if args.reference:
            if sha(args.reference / 'result.json') != args.reference_sha256:
                raise ValueError('reference receipt changed')
            reference = read(args.reference / 'result.json')
            if (reference['status'] != 'PASS_FROZEN_EVENT_ENCODERS'
                    or reference['phase'] != args.phase or reference['pins'] != pins):
                raise ValueError('complete same-phase/source/input reference required')
        result.update(pins=pins, external_source_revision=revision, dtype='float32',
                      input_receipt_sha256=INPUT_SHA, reference_sha256=args.reference_sha256)
        if args.phase == 'audio':
            import torch
            from safetensors.torch import load_file
            from transformers import WhisperConfig, WhisperFeatureExtractor
            from transformers.models.whisper.modeling_whisper import WhisperEncoder

            from scripts.capture_native_language import state_digest

            torch.set_num_threads(4)
            torch.set_num_interop_threads(1)
            torch.manual_seed(17)
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            if not torch.cuda.is_available():
                raise RuntimeError('registered local CUDA unavailable')
            torch.cuda.reset_peak_memory_stats()
            config = WhisperConfig.from_pretrained(speech_path, local_files_only=True)
            config._attn_implementation = 'eager'
            speech = WhisperEncoder(config)
            speech.load_state_dict(load_file(speech_path / 'encoder.safetensors'), strict=True)
            speech.eval().requires_grad_(False).to('cuda')
            processor = WhisperFeatureExtractor.from_pretrained(speech_path, local_files_only=True)
            sys.path.insert(0, str(source))
            from models.dymn.model import get_model
            from models.preprocess import AugmentMelSTFT

            with contextlib.redirect_stdout(io.StringIO()):
                event = get_model(width_mult=.4, pretrained_name=None, T_max=1.)
                mel = AugmentMelSTFT()
            event.load_state_dict(torch.load(audio_config['event_weights'], weights_only=True,
                                            map_location='cpu'), strict=True)
            event.eval().requires_grad_(False).to('cuda')
            mel.eval().requires_grad_(False).to('cuda')
            models = dict(speech=speech, event=event, mel=mel)
            result['parameters'] = {k: sum(p.numel() for p in v.parameters())
                                     for k, v in models.items()}
            if result['parameters'] != dict(speech=8208384, event=1961239, mel=0):
                raise ValueError('audio model inventory changed')
            def digest():
                return {k: state_digest(v) for k, v in models.items()}
            result.update(device=torch.cuda.get_device_name(), cuda=torch.version.cuda,
                attention='eager', speech_revision=audio_config['speech_revision'],
                versions={p: importlib.metadata.version(p) for p in
                          ('torch', 'torchaudio', 'transformers', 'numpy', 'safetensors')})
        else:
            if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
                raise ValueError('isolated CPU-only video process required')
            import jax
            import jax.numpy as jnp
            import tensorflow as tf
            from videoprism import models as vp

            from scripts.extract_frozen_videoprism_features import _preprocess
            from scripts.native_full_extent import encoder_distribution_versions
            from tiny_omni_decision.video_token_grid import token_grid

            tf.config.threading.set_intra_op_parallelism_threads(4)
            tf.config.threading.set_inter_op_parallelism_threads(2)
            if any(d.platform != 'cpu' for d in jax.devices()):
                raise ValueError('CPU-only VideoPrism required')
            model = vp.get_model('videoprism_public_v1_base')
            variables = vp.load_pretrained_weights('videoprism_public_v1_base',
                                                   checkpoint_path=str(checkpoint))
            result['parameters'] = sum(int(v.size) for v in jax.tree_util.tree_leaves(variables))
            if result['parameters'] != 114365184:
                raise ValueError('video model inventory changed')

            def digest():
                h = hashlib.sha256()
                for v in jax.tree_util.tree_leaves(variables):
                    x = np.asarray(v)
                    h.update(str(x.shape).encode())
                    h.update(x.tobytes())
                return h.hexdigest()

            encode = jax.jit(lambda x: model.apply(variables, x, train=False))
            result.update(device='cpu', versions=encoder_distribution_versions(),
                geometry=dict(raw=[8, 256, 456, 3], resized=[8, 288, 513, 3],
                              crop_left=112, crop_right=113, output=[8, 288, 288, 3],
                              policy='historical bilinear short-side288 center-crop divide255'))
        before = digest()
        result['frozen_before'] = before
        result['load_seconds'] = time.perf_counter() - started
        write(args.output / 'result.json', result)
        for row in inputs['rows']:
            guard()
            ident = row['parent']
            position = event_positions(np.load(INPUT / (ident + '-video-pts.npy')),
                                       row['audio']['start_seconds'], row['audio']['end_seconds'])
            tick = time.perf_counter()
            if args.phase == 'audio':
                acoustic, events = prepared_audio_tokens(speech, processor, event, mel,
                    np.load(INPUT / (ident + '-direct-16000.npy'), allow_pickle=False),
                    np.load(INPUT / (ident + '-direct-32000.npy'), allow_pickle=False))
                torch.cuda.synchronize()
                arrays = dict(speech=acoustic.cpu().numpy(), event_audio=events.cpu().numpy())
            else:
                rgb = np.load(INPUT / (ident + '-rgb.npy'), allow_pickle=False)
                if rgb.shape != (8, 256, 456, 3) or rgb.dtype != np.uint8:
                    raise ValueError('exact original eight RGB frames required')
                prepared = _preprocess(rgb, tf)
                save(ident + '-encoder-input', prepared)
                output = encode(jnp.asarray(prepared[None]))
                jax.block_until_ready(output)
                value = output[0] if isinstance(output, tuple) else output
                arrays = dict(video=token_grid(np.asarray(value)).reshape(2048, 768))
            elapsed = time.perf_counter() - tick
            for kind, value in arrays.items():
                save(ident + '-' + kind, value)
                save(ident + '-' + kind + '-positions', position[kind])
            result['rows'].append(dict(parent=ident, event_row=0, forward_seconds=elapsed,
                                       shapes={k: list(v.shape) for k, v in arrays.items()}))
            print(json.dumps(result['rows'][-1]), flush=True)
            write(args.output / 'result.json', result)
        result['frozen_after'] = digest()
        if before != result['frozen_after']:
            raise ValueError('frozen encoder state changed')
        if args.phase == 'audio':
            result['peak_vram_bytes'] = torch.cuda.max_memory_allocated()
            if result['peak_vram_bytes'] > 2 * 2**30:
                raise RuntimeError('fixed2GiB VRAM guard exceeded')
        if reference and (reference['versions'] != result['versions']
                          or reference['frozen_before'] != before
                          or reference['artifacts'] != artifacts):
            raise ValueError('replay runtime, state or complete artifact set differs')
        check_source(source, revision, package)
        for path, expected in pins.items():
            if sha(Path(path)) != expected:
                raise ValueError('source/input changed during probe')
        guard()
        result['status'] = 'PASS_FROZEN_EVENT_ENCODERS'
    except BaseException as exc:
        result.update(status='INCOMPLETE', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        result.update(seconds=time.perf_counter() - started,
                      finished_utc=datetime.now(UTC).isoformat(),
                      artifacts=artifacts)
        write(args.output / 'result.json', result)
        print(result['status'], flush=True)


if __name__ == '__main__':
    main()
