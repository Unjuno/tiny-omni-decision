"""Apply the fixed Lombard video head to GRID smoke features without scoring."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.audit_droid_train_temporal import digest
from scripts.lombard_temporal_model import TemporalSlotHead

ROOT = Path('C:/CodexArtifacts/pretrained-reuse-r2')
FEATURES = ROOT / 'grid-s1-native-video-20261028-attempt-01'
HEAD = ROOT / 'lombard-temporal-full-01/video_ordered'
PINS = {FEATURES / 'summary.json': (
            '41f1af1e0213179d6cec7249f02532a2aa34705feb50836a4bfc74ee362b73d9'),
        HEAD / 'best.pt': '4261a296711b0cfac2a7681bc250ddc4c2d59d8265119f0143e798e2936397d9',
        HEAD / 'normalizer.npz': '13e3c7367c9c0a079c29f4acdff5ce4263883cfaeffd1714f646210b0698c289',
        Path('scripts/lombard_temporal_model.py'): (
            'ba0e87fdacd9a465cbbac6557b0c66f333752f6fb6929123a53258546b56daee')}
REPO = Path(__file__).resolve().parents[1]


def valid_logits(value):
    array = np.asarray(value)
    if array.shape != (16, 51) or not np.isfinite(array).all():
        raise ValueError('invalid logits')
    return array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    out = parser.parse_args().output_dir.resolve()
    if out.exists() or out.is_relative_to(REPO) or any(digest(p) != h for p, h in PINS.items()):
        raise ValueError('output or pinned input mismatch')
    source = json.loads((FEATURES / 'summary.json').read_bytes())
    if len(source['calls']) != 16:
        raise ValueError('unexpected feature count')
    values = []
    for index, item in enumerate(source['calls'], 1):
        feature = np.load(FEATURES / 'worker' / f'call-{index:02}.npy', allow_pickle=False)
        if (feature.shape != (16, 768) or not np.isfinite(feature).all()
                or hashlib.sha256(feature.tobytes()).hexdigest() != item['feature_sha256']):
            raise ValueError('feature integrity mismatch')
        values.append(feature)
    model = TemporalSlotHead().eval().requires_grad_(False)
    model.load_state_dict(torch.load(HEAD / 'best.pt', map_location='cpu', weights_only=True),
                          strict=True)
    if sum(p.numel() for p in model.parameters()) != 106355:
        raise ValueError('unexpected head size')
    with np.load(HEAD / 'normalizer.npz', allow_pickle=False) as data:
        video = (np.asarray(values) - data['video_mean']) / data['video_scale']
    if not np.isfinite(video).all():
        raise ValueError('nonfinite normalized features')
    audio = torch.zeros((16, 384), dtype=torch.float32)
    present = torch.tensor([[0.0, 1.0]] * 16, dtype=torch.float32)
    with torch.inference_mode():
        raw = model(audio, torch.from_numpy(video.astype(np.float32)), present).numpy()
        logits = valid_logits(raw)
    if any(digest(p) != h for p, h in PINS.items()):
        raise ValueError('input changed during execution')
    out.mkdir(parents=True)
    np.save(out / 'video_logits.npy', logits)
    report = {'count': len(logits), 'head_parameters': 106355, 'training': False,
              'quality_evaluation': False, 'audio_available': False,
              'logits_sha256': digest(out / 'video_logits.npy'),
              'source_hashes': {str(k): v for k, v in PINS.items()},
              'script_sha256': digest(Path(__file__))}
    (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
