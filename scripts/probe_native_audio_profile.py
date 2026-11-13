"""Two fixed TRAIN inputs through both audio encoders; no fit or quality scoring."""

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import UTC, datetime
from pathlib import Path

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['HF_HUB_OFFLINE'] = '1'

import numpy as np
import psutil
import soundfile as sf
import torch
from safetensors.torch import load_file
from transformers import WhisperConfig, WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from scripts.capture_native_language import state_digest, verify, write
from scripts.native_candidate_inputs import require_source_revision
from scripts.native_video_cache import sha
from scripts.starss23_stream import audio_windows
from tiny_omni_decision.native_audio import audio_tokens

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('C:/CodexArtifacts/pretrained-reuse-r2')
CONFIG = REPO / 'configs/pretrained_reuse/native_audio_profile_v1.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--reference-sha256')
    args = parser.parse_args()
    if bool(args.reference) != bool(args.reference_sha256):
        raise ValueError('paired reference/hash required')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO).strip():
        raise ValueError('clean committed source required')
    args.output.mkdir(exist_ok=False)
    tick = time.perf_counter()
    report = dict(status='RUNNING', started_utc=datetime.now(UTC).isoformat(), command=sys.argv,
                  source_commit=subprocess.check_output(
                      ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                  optimizer_updates=0, quality_evaluation=False, calibration_loaded=False,
                  validation_loaded=False, sealed_audit_loaded=False, cloud_cost=0)
    write(args.output / 'result.json', report)
    try:
        config = json.loads(CONFIG.read_text(encoding='utf-8'))
        pins = config['input_pins'].copy()
        paths = subprocess.check_output(['git', 'ls-files', '-z', '--', '*.py'], cwd=REPO,
                                        text=True).split('\0')
        pins.update({str(REPO / p): sha(REPO / p) for p in paths if p})
        pins[str(CONFIG)] = sha(CONFIG)
        source = Path(config['event_source'])
        require_source_revision(source, config['event_revision'], 'models')
        pins.update({str(p): sha(p) for p in source.rglob('*.py')})
        verify(pins)
        report.update(validated_hashes=pins, config_sha256=sha(CONFIG))
        write(args.output / 'effective-config.json', config)
        if shutil.disk_usage(args.output).free < 2 * 2**30:
            raise RuntimeError('2GiB disk reserve required')
        reference = None
        if args.reference:
            if sha(args.reference / 'result.json') != args.reference_sha256:
                raise ValueError('reference hash differs')
            reference = json.loads((args.reference / 'result.json').read_text(encoding='utf-8'))
            if (reference['status'] != 'PASS_NATIVE_AUDIO_PROFILE'
                    or reference['validated_hashes'] != pins):
                raise ValueError('complete unchanged reference required')
            verify({str(args.reference / k): v for k, v in reference['artifact_hashes'].items()})
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.manual_seed(17)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        if not torch.cuda.is_available():
            raise RuntimeError('registered CUDA profile unavailable')
        torch.cuda.reset_peak_memory_stats()
        speech_path = Path(config['speech_model'])
        speech_config = WhisperConfig.from_pretrained(speech_path, local_files_only=True)
        speech_config._attn_implementation = 'eager'
        speech = WhisperEncoder(speech_config)
        speech.load_state_dict(load_file(speech_path / 'encoder.safetensors'), strict=True)
        speech.eval().requires_grad_(False).to('cuda')
        processor = WhisperFeatureExtractor.from_pretrained(speech_path, local_files_only=True)
        sys.path.insert(0, str(source))
        from models.dymn.model import get_model
        from models.preprocess import AugmentMelSTFT

        with contextlib.redirect_stdout(io.StringIO()):
            event = get_model(width_mult=.4, pretrained_name=None, T_max=1.)
            mel = AugmentMelSTFT()
        event.load_state_dict(torch.load(config['event_weights'], weights_only=True,
                                        map_location='cpu'), strict=True)
        event.eval().requires_grad_(False).to('cuda')
        mel.eval().requires_grad_(False).to('cuda')
        models = dict(speech=speech, event=event, mel=mel)
        before = {k: state_digest(v) for k, v in models.items()}
        report['parameters'] = {k: sum(p.numel() for p in v.parameters())
                                 for k, v in models.items()}
        if report['parameters'] != dict(speech=8208384, event=1961239, mel=0):
            raise ValueError('audio parameter inventory changed')
        with zipfile.ZipFile(config['lombard_archive']) as archive:
            content = archive.read(config['lombard_member'])
        if hashlib.sha256(content).hexdigest() != config['lombard_sha256']:
            raise ValueError('Lombard waveform differs')
        pcm, rate = sf.read(io.BytesIO(content), dtype='float32')
        if rate != 16000 or pcm.shape != (35200,):
            raise ValueError('Lombard native reference identity differs')
        inputs = [('lombard', pcm, rate)]
        with Path(config['starss_manifest']).open(encoding='utf-8') as handle:
            row = json.loads(next(handle))
        if (row['split'] != 'train' or row['id'] != config['starss_id']
                or row['media']['audio']['sha256'] != config['starss_audio_sha256']):
            raise ValueError('fixed TRAIN STARSS identity differs')
        frames = next(audio_windows(row['media']['audio']['path'],
                                   [(row['start_seconds'], row['end_seconds'])]))
        inputs.append(('starss', frames[:, 0].copy(), 24000))
        report['rows'] = []
        for name, waveform, rate in inputs:
            acoustic, event_features = audio_tokens(speech, processor, event, mel,
                                                     torch.from_numpy(waveform), rate)
            values = {'pcm': waveform, 'speech': acoustic.cpu().numpy(),
                      'event': event_features.cpu().numpy()}
            deltas = {}
            for kind, value in values.items():
                filename = name + '-' + kind + '.npy'
                with (args.output / filename).open('xb') as handle:
                    np.save(handle, value, allow_pickle=False)
                if not np.array_equal(np.load(args.output / filename, allow_pickle=False), value):
                    raise ValueError('save/reload mismatch')
                if reference:
                    previous = np.load(args.reference / filename, allow_pickle=False)
                    deltas[kind] = float(np.max(np.abs(previous - value)))
                    if not np.array_equal(previous, value):
                        raise ValueError('fresh audio replay differs')
                if name == 'lombard' and kind in {'speech', 'event'}:
                    previous = np.load(config['native_reference'][kind], allow_pickle=False)
                    deltas['native_' + kind] = float(np.max(np.abs(previous - value)))
                    if not np.array_equal(previous, value):
                        raise ValueError('original integrated audio frontend differs')
            report['rows'].append(dict(source=name, sample_rate=rate, samples=len(waveform),
                shapes={k: list(v.shape) for k, v in values.items()}, deltas=deltas))
        after = {k: state_digest(v) for k, v in models.items()}
        if before != after:
            raise ValueError('frozen audio models changed')
        report.update(frozen_before=before, frozen_after=after, dtype='float32', attention='eager',
            torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
            versions={p: importlib.metadata.version(p) for p in (
                'torch', 'torchaudio', 'transformers', 'numpy', 'soundfile', 'safetensors')},
            peak_vram_bytes=torch.cuda.max_memory_allocated(),
            os_peak_working_set_bytes=psutil.Process().memory_info().peak_wset)
        verify(pins)
        report['artifact_hashes'] = {p.name: sha(p) for p in args.output.iterdir()
                                     if p.is_file() and p.name != 'result.json'}
        if (time.perf_counter() - tick > 300 or shutil.disk_usage(args.output).free < 2 * 2**30
                or report['peak_vram_bytes'] > 2 * 2**30
                or report['os_peak_working_set_bytes'] > 4 * 2**30):
            raise RuntimeError('fixed300s/2GiB VRAM/4GiB host/2GiB free resource gate')
        report.update(status='PASS_NATIVE_AUDIO_PROFILE',
                      finished_utc=datetime.now(UTC).isoformat(),
                      seconds=time.perf_counter()-tick)
        write(args.output / 'result.json', report)
        print(report['status'], report['seconds'], flush=True)
    except BaseException as error:
        report.update(status='INCOMPLETE', error=repr(error), traceback=traceback.format_exc(),
                      finished_utc=datetime.now(UTC).isoformat(), seconds=time.perf_counter()-tick)
        write(args.output / 'result.json', report)
        raise


if __name__ == '__main__':
    main()
