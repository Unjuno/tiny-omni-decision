"""Exercise fixed real-media AV JSON interface on deliberately unmatched sources."""

import argparse
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import av
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import WhisperConfig, WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from scripts.audit_droid_train_temporal import digest
from scripts.lombard_temporal_model import TemporalSlotHead
from tiny_omni_decision.native_av_correspondence import NativeAVCorrespondence

ROOT = Path('C:/CodexArtifacts/pretrained-reuse-r2')
SPEECH = Path('C:/CodexArtifacts/pretrained-reuse-r1/whisper-encoder-runtime-01')
TRAIN = ROOT / 'lombard-grid-layout-02-media-gated/train.json'
AUDIO = ROOT / 'lombard-grid-archives-01/audio.zip'
HEAD = ROOT / 'lombard-temporal-full-01/audio'
VIDEO = ROOT / 'grid-s1-native-video-20261028-attempt-01'
VIDEO_HEAD = ROOT / 'grid-s1-video-head-20261028-attempt-01'
PARENT = ROOT / 'lombard-correspondence-02/result.json'
PINS = {
    TRAIN: '259f3ee36aa956618a89819338c2ecb67b194c82dbf636df70b551e00a5d78b0',
    AUDIO: 'e36707602fdbeaa2e2284acb29bba883261d5e86857f694ac40cbb56cdcc5e10',
    HEAD / 'best.pt': 'ade277dc1adf24bd621c838ebb8be8a7cbf26380889d45a3893400b4723a491a',
    HEAD / 'normalizer.npz': '13e3c7367c9c0a079c29f4acdff5ce4263883cfaeffd1714f646210b0698c289',
    SPEECH / 'encoder.safetensors': (
        'd7f4b75d10f8ec4af79a32cc6dd20160f0c669c67bd104c1f1ab3f9d5d36a2a8'),
    SPEECH / 'config.json': 'd8c8d403350cf4e21b4351d3812fc61defc6c5784d2d8f8a622d4b8ca23b98e9',
    SPEECH / 'preprocessor_config.json': (
        '3e4dc53c73e7e3954fc4435034f5027733c61d7c123b38bbd1461598fd8503ef'),
    VIDEO / 'summary.json': '41f1af1e0213179d6cec7249f02532a2aa34705feb50836a4bfc74ee362b73d9',
    VIDEO_HEAD / 'video_logits.npy': (
        '7cc918e3c36e39fd29da10022ae4b0e1174e22b1e5f729a9d2cd3532afd39bea'),
    PARENT: '9638d06b7fe4c0e3c53e0a0080d38b350eb135fd8cf819e2e3fcfdbf8e977516',
    ROOT / 'grid-cross-source-json-20261028-attempt-01/summary.json': (
        'f33039d4e58a612051e7d359619aaf031319693311bd4daee083b4c194de052f'),
    Path('scripts/lombard_temporal_model.py'): (
        'ba0e87fdacd9a465cbbac6557b0c66f333752f6fb6929123a53258546b56daee'),
    Path('src/tiny_omni_decision/native_av_correspondence.py'): (
        'd18eb335082f8d6a326b07c5ecfbc22e83b059c6e3a4b4b55b45bf9fab2f7311'),
}
REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / 'docs/pretrained_reuse/R2_GRID_CROSS_SOURCE_CONTROLS_PROTOCOL_20261028.md'


def validate_json(payload, option_ids):
    result = json.loads(payload)
    values = result['probabilities']
    if (set(values) != set(option_ids) or result['selected_option_id'] not in option_ids
            or not all(np.isfinite(x) and 0 <= x <= 1 for x in values.values())
            or abs(sum(values.values()) - 1) > 1e-6):
        raise ValueError('invalid output probabilities')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    out = parser.parse_args().output_dir.resolve()
    if out.exists() or out.is_relative_to(REPO) or any(digest(p) != h for p, h in PINS.items()):
        raise ValueError('output or pinned input mismatch')
    start = time.perf_counter()
    rows = json.loads(TRAIN.read_bytes())
    with zipfile.ZipFile(AUDIO) as archive:
        audio = archive.read(rows[0]['assets']['audio']['name'])
    audio_sha = hashlib.sha256(audio).hexdigest()
    calls = json.loads((VIDEO / 'summary.json').read_bytes())['calls'][:4]
    logits = np.load(VIDEO_HEAD / 'video_logits.npy', allow_pickle=False)[:4]
    if logits.shape != (4, 51) or not np.isfinite(logits).all():
        raise ValueError('invalid cached video logits')
    videos = [(VIDEO / 'media' / (item['media_sha256'] + '.mpg')).read_bytes() for item in calls]
    if any(hashlib.sha256(v).hexdigest() != item['media_sha256']
           for v, item in zip(videos, calls, strict=True)):
        raise ValueError('video media hash mismatch')
    torch.set_num_threads(4)
    encoder = WhisperEncoder(WhisperConfig.from_pretrained(SPEECH, local_files_only=True))
    encoder.load_state_dict(load_file(SPEECH / 'encoder.safetensors'), strict=True)
    encoder.eval().requires_grad_(False)
    if sum(p.numel() for p in encoder.parameters()) != 8208384:
        raise ValueError('unexpected encoder size')
    processor = WhisperFeatureExtractor.from_pretrained(SPEECH, local_files_only=True)
    model = TemporalSlotHead().eval().requires_grad_(False)
    model.load_state_dict(torch.load(HEAD / 'best.pt', map_location='cpu', weights_only=True),
                          strict=True)
    with np.load(HEAD / 'normalizer.npz', allow_pickle=False) as normalizer:
        mean, scale = normalizer['audio_mean'].copy(), normalizer['audio_scale'].copy()
    with av.open(io.BytesIO(audio)) as container:
        stream = container.streams.audio[0]
        if stream.codec_context.sample_rate != 16000:
            raise ValueError('unexpected Lombard sample rate')
        resampler = av.AudioResampler(format='fltp', layout='mono', rate=16000)
        chunks = []
        for frame in container.decode(stream):
            chunks.extend(v.to_ndarray() for v in resampler.resample(frame))
        chunks.extend(v.to_ndarray() for v in resampler.resample(None))
        wave = np.concatenate(chunks, axis=1)[0]
    if not 0 < len(wave) <= 480000 or not np.isfinite(wave).all():
        raise ValueError('invalid audio waveform')
    inputs = processor(wave, sampling_rate=16000, return_tensors='pt').input_features
    with torch.inference_mode():
        feature = encoder(inputs).last_hidden_state[0, :(len(wave) + 319) // 320].numpy().mean(0)
        normalized = torch.from_numpy(((feature - mean) / scale)[None])
        audio_logits = model(normalized, torch.zeros((1, 16, 768)),
                             torch.tensor([[1.0, 0.0]])).numpy()[0]
    if audio_logits.shape != (51,) or not np.isfinite(audio_logits).all():
        raise ValueError('invalid audio logits')
    video_lookup = {hashlib.sha256(v).hexdigest(): logit
                    for v, logit in zip(videos, logits, strict=True)}
    parent = json.loads(PARENT.read_bytes())
    runtime = NativeAVCorrespondence(lambda _: audio_logits,
                                     lambda b: video_lookup[hashlib.sha256(b).hexdigest()],
                                     parent['models']['audio']['temperatures'],
                                     parent['models']['video_ordered']['temperatures'],
                                     parent['query_temperature'], '0' * 64)
    a = runtime.encode('audio', audio, time_range=(0, len(wave) / 16000), expected_sha256=audio_sha)
    vs = [runtime.encode('video', v, time_range=(0, 2.96)) for v in videos]
    ids = [hashlib.sha256(f'grid-cross-source-smoke-{i}'.encode()).hexdigest() for i in range(4)]
    decision = validate_json(runtime.decide(a, vs, ids), ids)
    previous_path = ROOT / 'grid-cross-source-json-20261028-attempt-01/summary.json'
    previous = json.loads(previous_path.read_bytes())
    if decision != previous['decision']:
        raise ValueError('baseline decision did not reproduce')
    permutation = [2, 0, 3, 1]
    reordered = validate_json(runtime.decide(a, [vs[i] for i in permutation],
                                             [ids[i] for i in permutation]), ids)
    if reordered != decision:
        raise ValueError('candidate reorder changed result')
    without_audio = validate_json(runtime.decide(None, vs, ids), ids)
    without_video = validate_json(runtime.decide(a, None, ids), ids)
    if any(p != 0.25 for result in (without_audio, without_video)
           for p in result['probabilities'].values()):
        raise ValueError('missing-evidence policy differs from uniform')
    swapped = validate_json(runtime.decide(a, [vs[1], vs[0], vs[2], vs[3]], ids), ids)
    swap_l1 = sum(abs(swapped['probabilities'][i] - decision['probabilities'][i]) for i in ids)
    if runtime.encode('audio', audio, time_range=(0, len(wave) / 16000)) is not a:
        raise ValueError('audio state cache mismatch')
    if any(digest(p) != h for p, h in PINS.items()):
        raise ValueError('pinned input changed')
    out.mkdir(parents=True)
    report = {'scope': 'unmatched cross-source interface controls', 'decision': decision,
              'reordered': reordered, 'without_audio': without_audio,
              'without_video': without_video, 'swapped': swapped, 'swap_l1': swap_l1,
              'audio_sha256': audio_sha, 'video_sha256': [c['media_sha256'] for c in calls],
              'audio_samples': len(wave), 'audio_encoder_parameters': 8208384,
              'seconds': time.perf_counter() - start, 'backend_calls': runtime.backend_calls,
              'cache_hits': runtime.cache_hits,
              'source_hashes': {str(k): v for k, v in PINS.items()},
              'protocol_sha256': digest(PROTOCOL), 'script_sha256': digest(Path(__file__)),
              'gpu_used': False, 'training': False, 'quality_evaluation': False}
    (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
