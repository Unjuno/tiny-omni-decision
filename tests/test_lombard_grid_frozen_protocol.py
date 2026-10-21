"""Portable checks of the pre-quality native AV protocol and locally pinned source."""

import hashlib
import json
from pathlib import Path

from scripts.lombard_grid_corpus import SLOTS
from tiny_omni_decision.dataset import audit_license

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / 'configs/pretrained_reuse/lombard-grid-pooled-slot-v1.json'
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding='utf-8'))


def test_native_source_snapshot_is_allow_and_not_a_fabricated_upstream_revision():
    raw = (ROOT / PROTOCOL['source_manifest']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PROTOCOL['source_manifest_sha256']
    source = json.loads(raw)
    assert 'revision' not in source
    assert source['source_revision_type'].startswith('locally hashed')
    policy, unresolved = audit_license(
        source['license'], has_media=True,
        **{key: source[key] for key in ['commercial_use', 'derivative_model_training_allowed',
                                       'redistribution_allowed', 'media_redistribution_allowed',
                                       'trust_status']},
    )
    assert policy == source['policy'] == 'ALLOW' and not unresolved
    assert source['train'] == 4080 and source['calibration'] == 398 and source['validation'] == 889
    assert source['unresolved_labels_retained'] == 19
    assert source['broken_training_media_retained'] == 4
    assert source['split_hashes'] == PROTOCOL['split_hashes']
    for info in source['archives'].values():
        assert info['status'] == 'PASS' and len(info['sha256']) == 64
        assert info['all_member_crc_verified'] and info['members'] == 5390


def test_every_native_option_including_all_25_letters_is_preserved():
    expected = {slot: list(table.values()) for slot, table in SLOTS.items()}
    assert PROTOCOL['options'] == expected
    assert [len(v) for v in PROTOCOL['options'].values()] == [4, 4, 4, 25, 10, 4]
    assert sum(len(v) for v in PROTOCOL['options'].values()) == 51
    assert PROTOCOL['input_dimensions'] == 384 + 768 + 2
    expected_parameters = (PROTOCOL['input_dimensions'] + 1) * 64 + (64 + 1) * 51
    assert PROTOCOL['head_parameters'] == expected_parameters == 77235


def test_three_arms_budget_selector_calibration_are_fixed_before_quality():
    assert PROTOCOL['status'] == 'FROZEN_BEFORE_ANY_READOUT_QUALITY_METRICS'
    assert PROTOCOL['arms'] == ['audio', 'video', 'joint']
    assert (PROTOCOL['seed'], PROTOCOL['epochs'], PROTOCOL['batch_size']) == (17, 24, 128)
    assert PROTOCOL['learning_rate'] == .001 and PROTOCOL['weight_decay'] == .001
    assert not PROTOCOL['encoder_training']
    assert 'raw validation macro NLL' in PROTOCOL['selector']
    assert 'independent calibration split' in PROTOCOL['calibration']
    assert 'no extra seed/epoch/policy search' in PROTOCOL['stopping']
