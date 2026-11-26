"""Fail closed on source/identity/feature mismatch before any native slot training."""

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.lombard_grid_corpus import decode_code
from scripts.lombard_readout_core import OPTIONS, targets_from_rows
from tiny_omni_decision.dataset import audit_license

REPO = Path(__file__).resolve().parents[1]
PROTOCOL_SHA = '047e98485298c5ff26900e20335aa49a1aa8cbf0581772482edc2dad3c19afe4'
SOURCE_REVISION = 'c6eba0ece3dd49a24ffc44582966997a4a59f1d8'
CHECKPOINT_SHA = '61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29'
MEDIA_SHA = '3a29c73421bb695b9179e540ade00484a2ab6e744f90d1a62b078797048a054b'
AUDIO_REPORT_SHA = 'd72ae9cd299419feecc307d00dc30bd26fad9907a04ee4ffaf82d31a361c8bed'
VIDEO_SMOKE_SHA = '3d1a4c45f4d191b314af4ba0950a4fd7d819ff5afa3c714ec941d37f2f5f2ae6'
VIDEO_RUNNER_SHA = '965099b6b9bef40dd033aaa862f66725bcb2694fddab889aff970edf88ad558d'
SMOKE_IDS = {'s16_l_bgah2n', 's16_p_bgah2n', 's22_l_sgbe5s', 's22_p_sgbe5s'}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path, expected=None):
    if not path.exists():
        raise ValueError(f'required completed artifact absent: {path}')
    if expected is not None and sha(path) != expected:
        raise ValueError(f'artifact hash differs: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def validate_bank(info, values, expected_rows, kind, media_rows):
    if kind not in {'audio', 'video'} or info.get('status') != 'PASS':
        raise ValueError('complete PASS banks required; allocated files are insufficient')
    width = 384 if kind == 'audio' else 768
    if (values.shape != (len(expected_rows), width) or values.dtype != np.float32
            or not np.isfinite(values).all()
            or (info.get('dtype') if kind == 'audio' else info.get('config', {}).get('dtype'))
            != 'float32'
            or info.get('training') is not False):
        raise ValueError('native FP32 frozen bank shape/dtype/value differs')
    ids = [r['source_id'] for r in expected_rows]
    feature_rows = info.get('rows', [])
    if ([r['source_id'] for r in feature_rows] != ids or len(set(ids)) != len(ids)):
        raise ValueError('feature IDs/order/uniqueness differ from fixed corpus')
    offset = 0
    for feature, source in zip(feature_rows, expected_rows, strict=True):
        media = media_rows[source['source_id']]
        if (feature['split'] != source['split']
                or feature['input_sha256'] != media[kind + '_sha256']):
            raise ValueError('feature split/input media hash binding differs')
        if kind == 'audio':
            count = (media['samples'] + 319)//320
            if (feature['samples'] != media['samples'] or feature['offset'] != offset
                    or feature['tokens'] != count
                    or feature.get('repeat_bitwise_equal') is not True):
                raise ValueError('physical audio token accounting/repeat differs')
            offset += count
        else:
            indices, times = feature['selected_indices'], feature['selected_pts']
            if (len(indices) != 16 or len(times) != 16 or indices[0] != 0
                    or indices[-1] != media['video_frames']-1
                    or any(not isinstance(i, int) for i in indices)
                    or np.any(np.diff(indices) < 0) or not np.isfinite(times).all()
                    or np.any(np.diff(times) < 0) or times[0] != media['first_video_pts']
                    or times[-1] != media['last_video_pts']):
                raise ValueError('native chronological frame sampling differs')
    if kind == 'audio':
        if (info.get('model_parameters') != 8208384
                or info.get('all_parameters_unchanged') is not True
                or info.get('all_inputs_complete') is not True):
            raise ValueError('complete frozen native speech bank required')
    else:
        config = info.get('config', {})
        if (info.get('parameters') != 114365184 or info.get('parameters_unchanged') is not True
                or info.get('all_features_reload_verified') is not True
                or info.get('all_media_hashes_verified') is not True
                or info.get('all_expected_ids_present') is not True
                or info.get('parameter_sha256_before') != info.get('parameter_sha256_after')
                or len(info.get('parameter_sha256_before', '')) != 64
                or info.get('native_reference_checks') != 4
                or any(config.get(k) != v for k, v in dict(
                    frames=16, height=288, width=288, dtype='float32', training=False,
                    labels_transcripts_alignments_as_input=False,
                    source_revision=SOURCE_REVISION, checkpoint_sha256=CHECKPOINT_SHA,
                    total_rows=len(expected_rows)).items())):
            raise ValueError('complete unchanged native VideoPrism profile/reload required')


def read_bank(folder, expected_rows, kind, media_rows):
    info = read_json(folder/'result.json')
    if info.get('status') != 'PASS':
        raise ValueError('completed PASS bank required before reading allocated features')
    path = folder/'pooled.npy'
    if sha(path) != info['feature_hashes']['pooled.npy']:
        raise ValueError('pooled feature file hash differs')
    value = np.load(path, mmap_mode='r', allow_pickle=False)
    validate_bank(info, value, expected_rows, kind, media_rows)
    return value, info


def load_data(root, smoke=False):
    root = Path(root)
    protocol_path = REPO/'configs/pretrained_reuse/lombard-grid-pooled-slot-v1.json'
    protocol = read_json(protocol_path, PROTOCOL_SHA)
    source_path = REPO/protocol['source_manifest']
    source = read_json(source_path, protocol['source_manifest_sha256'])
    policy, unresolved = audit_license(source['license'], has_media=True, **{
        k: source[k] for k in ['commercial_use', 'derivative_model_training_allowed',
                              'redistribution_allowed', 'media_redistribution_allowed',
                              'trust_status']})
    if policy != 'ALLOW' or unresolved or protocol['options'] != OPTIONS:
        raise ValueError('explicit source rights and full native vocabularies required')
    layout = root/'lombard-grid-layout-02-media-gated'
    rows, owners = {}, {}
    for split, count in [('train', 4080), ('calibration', 398), ('validation', 889)]:
        rows[split] = read_json(layout/(split+'.json'), protocol['split_hashes'][split])
        if len(rows[split]) != count or any(r['split'] != split for r in rows[split]):
            raise ValueError('fixed full corpus counts/split differ')
        targets_from_rows(rows[split])
        for row in rows[split]:
            if row['targets'] != decode_code(row['actual_code']):
                raise ValueError('native labels differ from frozen actual transcription')
            for kind in ['source_id', 'speaker', 'speaker_group', 'actual_code']:
                key = (kind, row[kind])
                if key in owners and owners[key] != split:
                    raise ValueError('cross-split underlying source/content identity')
                owners[key] = split
            try:
                decode_code(row['intended_code'])
            except ValueError:
                continue  # only validated intended phrases are normalized-content identities
            key = ('actual_code', row['intended_code'])
            if key in owners and owners[key] != split:
                raise ValueError('cross-split intended/actual phrase identity')
            owners[key] = split
    expected = sum(rows.values(), [])
    if len({r['source_id'] for r in expected}) != 5367:
        raise ValueError('duplicate native source identity')
    media_dir = root/'lombard-grid-media-integrity-01'
    audit = read_json(media_dir/'result.json', MEDIA_SHA)
    media_list = read_json(media_dir/'rows.json', audit['rows_sha256'])
    media = {r['source_id']: r for r in media_list}
    if len(media) != len(media_list):
        raise ValueError('duplicate media audit identities')
    for row in expected:
        item = media[row['source_id']]
        if (item['audio_status'] != 'PASS' or item['video_status'] != 'PASS'
                or item['split'] != row['split']):
            raise ValueError('common valid media required for every comparison arm')
    for kind, pin in source['archives'].items():
        if sha(root/'lombard-grid-archives-01'/(kind+'.zip')) != pin['sha256']:
            raise ValueError('source archive hash differs')
    audio_dir = root/'lombard-grid-whisper-features-01'
    read_json(audio_dir/'result.json', AUDIO_REPORT_SHA)
    audio, audio_info = read_bank(audio_dir, expected, 'audio', media)
    if audio_info['split_hashes'] != protocol['split_hashes']:
        raise ValueError('speech bank split hashes differ')
    if smoke:
        video_dir = root/'lombard-grid-videoprism-native-features-smoke-02'
        read_json(video_dir/'result.json', VIDEO_SMOKE_SHA)
        subset = [r for r in rows['train'] if r['source_id'] in SMOKE_IDS]
        if len(subset) != 4:
            raise ValueError('all four predeclared training smoke sources required')
        video, video_info = read_bank(video_dir, subset, 'video', media)
        positions = {r['source_id']: i for i, r in enumerate(expected)}
        audio = audio[[positions[r['source_id']] for r in subset]]
        rows = {'train': subset}
    else:
        video_dir = root/'lombard-grid-videoprism-native-features-01'
        video, video_info = read_bank(video_dir, expected, 'video', media)
        config = read_json(video_dir/'effective-config.json', video_info['effective_config_sha256'])
        if (config != video_info['config'] or config['split_hashes'] != protocol['split_hashes']
                or video_info['runner_sha256'] != VIDEO_RUNNER_SHA
                or sha(video_dir/'runner.py') != VIDEO_RUNNER_SHA):
            raise ValueError('actual native extraction runner/config differs')
    raw, y, offset = {}, {}, 0
    for split, items in rows.items():
        count = len(items)
        raw[split] = (audio[offset:offset+count], video[offset:offset+count])
        y[split] = targets_from_rows(items)
        offset += count
    if offset != len(audio) or offset != len(video):
        raise ValueError('common feature/corpus accounting differs')
    return raw, y, rows, dict(
        protocol=protocol, protocol_sha256=PROTOCOL_SHA,
        source_manifest_sha256=sha(source_path), media_report_sha256=MEDIA_SHA,
        audio_report_sha256=sha(audio_dir/'result.json'),
        audio_features=audio_info['feature_hashes'],
        video_report_sha256=sha(video_dir/'result.json'),
        video_features=video_info['feature_hashes'],
        training_only_smoke=smoke,
    )
