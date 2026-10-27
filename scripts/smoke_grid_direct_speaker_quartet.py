"""Run one frozen, matched GRID quartet as an end-to-end technical smoke."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import numpy as np

from scripts.audit_droid_train_temporal import digest
from scripts.lombard_temporal_model import TemporalSlotHead
from tiny_omni_decision.native_av_correspondence import NativeAVCorrespondence

ROOT = Path('C:/CodexArtifacts/pretrained-reuse-r2')
SPEECH = Path('C:/CodexArtifacts/pretrained-reuse-r1/whisper-encoder-runtime-01')
AUDIO_HEAD = ROOT / 'lombard-temporal-full-01/audio'
VIDEO_HEAD = ROOT / 'lombard-temporal-full-01/video_ordered'
PARENT = ROOT / 'lombard-correspondence-02/result.json'
WORKER = Path('scripts/native_av_video_worker.py')
VIDEO_DEPS = Path('C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-r1/venv/Lib/site-packages')
VIDEO_SOURCE = Path('C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-r1/source')
PINS = {
    SPEECH / 'encoder.safetensors': (
        'd7f4b75d10f8ec4af79a32cc6dd20160f0c669c67bd104c1f1ab3f9d5d36a2a8'),
    SPEECH / 'config.json': 'd8c8d403350cf4e21b4351d3812fc61defc6c5784d2d8f8a622d4b8ca23b98e9',
    SPEECH / 'preprocessor_config.json': (
        '3e4dc53c73e7e3954fc4435034f5027733c61d7c123b38bbd1461598fd8503ef'),
    AUDIO_HEAD / 'best.pt': 'ade277dc1adf24bd621c838ebb8be8a7cbf26380889d45a3893400b4723a491a',
    AUDIO_HEAD / 'normalizer.npz': (
        '13e3c7367c9c0a079c29f4acdff5ce4263883cfaeffd1714f646210b0698c289'),
    VIDEO_HEAD / 'best.pt': '4261a296711b0cfac2a7681bc250ddc4c2d59d8265119f0143e798e2936397d9',
    VIDEO_HEAD / 'normalizer.npz': (
        '13e3c7367c9c0a079c29f4acdff5ce4263883cfaeffd1714f646210b0698c289'),
    PARENT: '9638d06b7fe4c0e3c53e0a0080d38b350eb135fd8cf819e2e3fcfdbf8e977516',
    Path('scripts/lombard_temporal_model.py'): (
        'ba0e87fdacd9a465cbbac6557b0c66f333752f6fb6929123a53258546b56daee'),
    Path('src/tiny_omni_decision/native_av_correspondence.py'): (
        'd18eb335082f8d6a326b07c5ecfbc22e83b059c6e3a4b4b55b45bf9fab2f7311'),
    WORKER: '5857a1d8973c7995c67efed448a87561e43a64ed122e49446d1d1e3f1a074093',
}
SELECTION_SHA256 = 'e897ff02a435a463211f0a157495058adcb5c390bf384437e3e56dc8066da281'
AUDIO_TAR_SHA256 = '9fafc520f218bc0e1b1b14e0b396c5470f4b0cb908d3712c3ca7b52de5d34792'
ALIGN_TAR_SHA256 = '7bee9636e68e34661dd8bef06d815e8ac7d31aa09a3b6ed7664fb0de3e337b94'
VIDEO_ZIP_SHA256 = '69e7c2ef14b1bd4f2d0012a8a13fe8e9375c4f11a03c4de9f03f4071f48f12fd'
REPO = Path(__file__).resolve().parents[1]


def validate_json(payload: str, option_ids: list[str]) -> dict:
    result = json.loads(payload)
    values = result['probabilities']
    if (set(values) != set(option_ids) or result['selected_option_id'] not in option_ids
            or not all(np.isfinite(x) and 0 <= x <= 1 for x in values.values())
            or abs(sum(values.values()) - 1) > 1e-6):
        raise ValueError('invalid output probabilities')
    return result


def read_audio_members(path: Path, speaker: str, codes: list[str]) -> dict[str, bytes]:
    wanted = {f'{speaker}/{code}.wav' for code in codes}
    found = {}
    with tarfile.open(path, 'r:*') as archive:
        for member in archive:
            if member.isfile() and member.name in wanted:
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError('audio member could not be read')
                found[Path(member.name).stem] = stream.read()
    if set(found) != set(codes):
        raise ValueError('selected audio codes are missing from direct speaker archive')
    return found


def read_video_members(path: Path, speaker: str, codes: list[str]) -> dict[str, bytes]:
    wanted = {f'{speaker}/{code}.mpg' for code in codes}
    found = {}
    with zipfile.ZipFile(path) as archive:
        for name in sorted(wanted):
            found[Path(name).stem] = archive.read(name)  # zipfile verifies member CRC
    return found


def decode_audio(data: bytes, processor) -> tuple[np.ndarray, float]:
    import av

    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.audio[0]
        if stream.codec_context.sample_rate != 25000:
            raise ValueError('direct GRID endpointed audio is not 25 kHz')
        resampler = av.AudioResampler(format='fltp', layout='mono', rate=16000)
        chunks = []
        for frame in container.decode(stream):
            chunks.extend(value.to_ndarray() for value in resampler.resample(frame))
        chunks.extend(value.to_ndarray() for value in resampler.resample(None))
    wave = np.concatenate(chunks, axis=1)[0]
    if not 0 < len(wave) <= 480000 or not np.isfinite(wave).all():
        raise ValueError('invalid resampled waveform')
    inputs = processor(wave, sampling_rate=16000, return_tensors='pt').input_features
    duration = len(wave) / 16000
    return inputs, duration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--audio-tar', type=Path, required=True)
    parser.add_argument('--alignment-tar', type=Path, required=True)
    parser.add_argument('--video-zip', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    inputs = {
        args.selection: SELECTION_SHA256,
        args.audio_tar: AUDIO_TAR_SHA256,
        args.alignment_tar: ALIGN_TAR_SHA256,
        args.video_zip: VIDEO_ZIP_SHA256,
        **PINS,
    }
    if out.exists() or out.is_relative_to(REPO) or any(digest(p) != h for p, h in inputs.items()):
        raise ValueError('new external output and every pinned input are required')
    rows = [json.loads(line) for line in args.selection.read_text(encoding='utf-8').splitlines()]
    row = rows[0]
    if row['speaker'] != 's30' or len(row['codes']) != 4 or len(row['queries']) != 4:
        raise ValueError('first frozen pilot row is not a balanced s30 quartet')
    if set(row['codes']) != {query['audio_code'] for query in row['queries']}:
        raise ValueError('quartet audio codes do not match its video candidates')

    started = time.perf_counter()
    out.mkdir(parents=True)
    media = out / 'media'
    media.mkdir()
    audio_bytes = read_audio_members(args.audio_tar, row['speaker'], row['codes'])
    video_bytes = read_video_members(args.video_zip, row['speaker'], row['codes'])
    audio_hashes = {code: hashlib.sha256(data).hexdigest()
                    for code, data in audio_bytes.items()}
    video_hashes = {code: hashlib.sha256(data).hexdigest()
                    for code, data in video_bytes.items()}
    for code, data in audio_bytes.items():
        (media / f'{audio_hashes[code]}.wav').write_bytes(data)
    for code, data in video_bytes.items():
        (media / f'{video_hashes[code]}.mpg').write_bytes(data)

    torch_threads = 4
    import torch
    from safetensors.torch import load_file
    from transformers import WhisperConfig, WhisperFeatureExtractor
    from transformers.models.whisper.modeling_whisper import WhisperEncoder

    torch.set_num_threads(torch_threads)
    encoder = WhisperEncoder(WhisperConfig.from_pretrained(SPEECH, local_files_only=True))
    encoder.load_state_dict(load_file(SPEECH / 'encoder.safetensors'), strict=True)
    encoder.eval().requires_grad_(False)
    if sum(p.numel() for p in encoder.parameters()) != 8_208_384:
        raise ValueError('unexpected frozen Whisper encoder size')
    processor = WhisperFeatureExtractor.from_pretrained(SPEECH, local_files_only=True)
    audio_model = TemporalSlotHead().eval().requires_grad_(False)
    audio_model.load_state_dict(torch.load(AUDIO_HEAD / 'best.pt', map_location='cpu',
                                          weights_only=True), strict=True)
    video_model = TemporalSlotHead().eval().requires_grad_(False)
    video_model.load_state_dict(torch.load(VIDEO_HEAD / 'best.pt', map_location='cpu',
                                          weights_only=True), strict=True)
    if sum(p.numel() for p in audio_model.parameters()) != 106_355:
        raise ValueError('unexpected frozen audio decision head size')
    if sum(p.numel() for p in video_model.parameters()) != 106_355:
        raise ValueError('unexpected frozen video decision head size')
    with np.load(AUDIO_HEAD / 'normalizer.npz', allow_pickle=False) as values:
        audio_mean, audio_scale = values['audio_mean'].copy(), values['audio_scale'].copy()
    with np.load(VIDEO_HEAD / 'normalizer.npz', allow_pickle=False) as values:
        video_mean, video_scale = values['video_mean'].copy(), values['video_scale'].copy()

    audio_logits = {}
    audio_durations = {}
    with torch.inference_mode():
        for code, data in audio_bytes.items():
            features, duration = decode_audio(data, processor)
            hidden = encoder(features).last_hidden_state[0]
            valid = max(1, min(hidden.shape[0], int((duration * 16000 + 319) // 320)))
            pooled = hidden[:valid].mean(0).numpy()
            normalized = torch.from_numpy(((pooled - audio_mean) / audio_scale)[None])
            value = audio_model(normalized, torch.zeros((1, 16, 768)),
                                torch.tensor([[1.0, 0.0]])).numpy()[0]
            if value.shape != (51,) or not np.isfinite(value).all():
                raise ValueError('invalid frozen audio logits')
            audio_logits[audio_hashes[code]] = value
            audio_durations[code] = duration

    worker_out = out / 'video-worker'
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(
        map(str, (VIDEO_DEPS, VIDEO_SOURCE, REPO, REPO / 'src')),
    )
    env['JAX_PLATFORMS'] = 'cpu'
    env['TF_CPP_MIN_LOG_LEVEL'] = '2'
    command = [sys.executable, '-m', 'scripts.native_av_video_worker',
               '--output', str(worker_out)]
    worker_start = time.perf_counter()
    with (out / 'video-worker.log').open('w', encoding='utf-8') as log:
        worker = subprocess.Popen(
            command, cwd=REPO, env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=log, text=True, encoding='utf-8',
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
        try:
            ready = json.loads(worker.stdout.readline())
            if not ready.get('ready'):
                raise ValueError('VideoPrism worker failed to initialize')
            video_features = {}
            metadata = []
            for code in row['codes']:
                video_path = media / f'{video_hashes[code]}.mpg'
                worker.stdin.write(json.dumps({
                    'path': str(video_path), 'sha256': video_hashes[code],
                }) + '\n')
                worker.stdin.flush()
                reply = json.loads(worker.stdout.readline())
                feature = np.asarray(reply['feature'], dtype=np.float32)
                if feature.shape != (16, 768) or not np.isfinite(feature).all():
                    raise ValueError('invalid frozen VideoPrism feature')
                video_features[code] = feature
                metadata.append(reply['metadata'])
                print(f'encoded_video={len(metadata)}/4', flush=True)
            worker.stdin.close()
            if worker.wait(timeout=300) != 0:
                raise RuntimeError('VideoPrism worker exited unsuccessfully')
        finally:
            if worker.poll() is None:
                worker.terminate()
                worker.wait(timeout=15)
    worker_report = json.loads((worker_out / 'result.json').read_bytes())
    if worker_report['status'] != 'PASS_NATIVE_VIDEO_WORKER' or len(metadata) != 4:
        raise ValueError('VideoPrism feature worker did not complete integrity checks')

    video_logits = {}
    with torch.inference_mode():
        for code, feature in video_features.items():
            normalized = torch.from_numpy(((feature - video_mean) / video_scale)[None])
            value = video_model(torch.zeros((1, 384)), normalized,
                                torch.tensor([[0.0, 1.0]])).numpy()[0]
            if value.shape != (51,) or not np.isfinite(value).all():
                raise ValueError('invalid frozen video logits')
            video_logits[video_hashes[code]] = value

    profile = hashlib.sha256(json.dumps(
        {str(path): sha for path, sha in inputs.items()}, sort_keys=True,
    ).encode()).hexdigest()
    calibration = json.loads(PARENT.read_bytes())
    runtime = NativeAVCorrespondence(
        lambda content: audio_logits[hashlib.sha256(content).hexdigest()],
        lambda content: video_logits[hashlib.sha256(content).hexdigest()],
        calibration['models']['audio']['temperatures'],
        calibration['models']['video_ordered']['temperatures'],
        calibration['query_temperature'], profile,
    )
    audio_states = {
        code: runtime.encode(
            'audio', data, time_range=(0.0, audio_durations[code]),
            expected_sha256=audio_hashes[code],
        ) for code, data in audio_bytes.items()
    }
    video_states = {
        code: runtime.encode(
            'video', data, time_range=(0.0, 2.96),
            expected_sha256=video_hashes[code],
        ) for code, data in video_bytes.items()
    }
    videos = [video_states[code] for code in row['codes']]
    decisions = []
    permutation = [2, 0, 3, 1]
    for query in row['queries']:
        audio = audio_states[query['audio_code']]
        option_ids = list(row['option_ids'])
        baseline = validate_json(runtime.decide(audio, videos, option_ids), option_ids)
        reordered = validate_json(
            runtime.decide(audio, [videos[i] for i in permutation],
                            [option_ids[i] for i in permutation]), option_ids,
        )
        without_audio = validate_json(runtime.decide(None, videos, option_ids), option_ids)
        without_video = validate_json(runtime.decide(audio, None, option_ids), option_ids)
        mismatched_audio = audio_states[next(
            code for code in row['codes'] if code != query['audio_code']
        )]
        audio_swap = validate_json(
            runtime.decide(mismatched_audio, videos, option_ids), option_ids,
        )
        video_swap = validate_json(
            runtime.decide(audio, videos[1:] + videos[:1], option_ids), option_ids,
        )
        if reordered != baseline:
            raise ValueError('reordering candidates together with IDs changed JSON')
        if any(value != 0.25 for result in (without_audio, without_video)
               for value in result['probabilities'].values()):
            raise ValueError('missing-input policy did not return uniform probabilities')
        decisions.append({
            'audio_code': query['audio_code'],
            'target_id': query['target_id'],
            'baseline': baseline,
            'reordered': reordered,
            'without_audio': without_audio,
            'without_video': without_video,
            'mismatched_audio': audio_swap,
            'mismatched_video': video_swap,
        })
    if any(digest(path) != expected for path, expected in inputs.items()):
        raise ValueError('a pinned input changed during the smoke')
    result = {
        'scope': 'one matched quartet technical smoke; not a quality evaluation',
        'quartet_id': row['quartet_id'],
        'speaker': row['speaker'],
        'query_count': len(decisions),
        'decisions': decisions,
        'audio_media_sha256': audio_hashes,
        'video_media_sha256': video_hashes,
        'audio_duration_seconds': audio_durations,
        'video_worker_result_sha256': digest(worker_out / 'result.json'),
        'video_worker_metadata': metadata,
        'video_worker_seconds': time.perf_counter() - worker_start,
        'total_seconds': time.perf_counter() - started,
        'backend_calls': runtime.backend_calls,
        'cache_hits': runtime.cache_hits,
        'cache_entries': runtime.cache_entries,
        'frozen_parameter_counts': {
            'whisper_encoder': 8_208_384,
            'audio_decision_head': 106_355,
            'video_decision_head': 106_355,
            'videoprism_base': 114_365_184,
        },
        'source_hashes': {str(path): expected for path, expected in inputs.items()},
        'profile_sha256': profile,
        'script_sha256': digest(Path(__file__)),
        'gpu_used': False,
        'training': False,
        'quality_evaluation': False,
        'sealed_audit_read': False,
        'interpretation': 'No accuracy aggregate or model-selection action was performed.',
    }
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(out), 'queries': len(decisions),
                      'seconds': result['total_seconds']}))


if __name__ == '__main__':
    main()
