"""Contract tests for the numerical policy; no optional ML imports required."""
from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import pytest


def policy_data(arm='q3_recovery_to_respecialization', profile='comparison'):
    fixed = arm == 'q2_fixed_recovery'
    return {'schema_version': 2,
         'kind': 'postquant_policy',
         'arm': arm,
         'profile': profile,
         'total_updates': 8 if profile == 'smoke' else 1024,
         'recovery_updates': 0 if fixed else 2 if profile == 'smoke' else 256,
         'gradient_accumulation_steps': 4,
         'temperature': 1.0,
         'full_vocab_kl': 0.0,
         'schedule': 'fixed' if fixed else 'recovery_then_linear',
         'recovery': {'option_kl': 1.0,
         'cross_entropy': 0.2,
         'brier': 0.2},
         'respecialization_end': {'option_kl': 1.0 if fixed else 0.2,
         'cross_entropy': 0.2 if fixed else 1.0,
         'brier': 0.2}}

def module():
    return importlib.import_module('tiny_omni_decision.postquant_policy')

def test_policy_module_exists():
    assert importlib.util.find_spec('tiny_omni_decision.postquant_policy') is not None

@pytest.mark.parametrize('profile,end,boundary', [('comparison', 1023, 256), ('smoke', 7, 2)])
def test_endpoints_and_control_match(profile, end, boundary):
    m = module()
    q3 = m.PostQuantConfig.model_validate(policy_data(profile=profile))
    q4 = m.PostQuantConfig.model_validate(policy_data('q4_high_precision_control', profile))
    values = [m.loss_weights_for_update(i, q3) for i in range(end + 1)]
    assert values == [m.loss_weights_for_update(i, q4) for i in range(end + 1)]
    for i in [0, boundary - 1, boundary]:
        assert values[i].option_kl == 1.0
        assert values[i].cross_entropy == 0.2
    assert values[boundary - 1].stage == 'recovery'
    assert values[boundary].stage == 'respecialization'
    assert values[-1].option_kl == 0.2
    assert values[-1].cross_entropy == 1.0
    assert all(a.option_kl >= b.option_kl for a, b in zip(values, values[1:], strict=False))
    assert all(a.cross_entropy <= b.cross_entropy for a,
         b in zip(values,
         values[1:],
         strict=False))
    assert all(v.brier == 0.2 for v in values)

def test_fixed_arm_never_anneals():
    m = module()
    c = m.PostQuantConfig.model_validate(policy_data('q2_fixed_recovery'))
    assert m.loss_weights_for_update(0, c) == m.loss_weights_for_update(1023, c)
    assert m.loss_weights_for_update(1023, c).stage == 'recovery'

@pytest.mark.parametrize('key,value',
     [('schema_version',
     1),
     ('schema_version',
     2.0),
     ('schema_version',
     True),
     ('total_updates',
     '1024'),
     ('total_updates',
     1024.0),
     ('total_updates',
     True),
     ('total_updates',
     0),
     ('recovery_updates',
     0),
     ('recovery_updates',
     1024),
     ('gradient_accumulation_steps',
     1),
     ('temperature',
     2.0),
     ('temperature',
     True),
     ('temperature',
     float('nan')),
     ('full_vocab_kl',
     0.1),
     ('arm',
     'q0_master'),
     ('schedule',
     'fixed'),
     ('profile',
     'unknown'),
     ('unknown_key',
     False)])
def test_invalid_policy_rejected(key, value):
    obj = policy_data()
    obj[key] = value
    with pytest.raises(ValueError):
        module().PostQuantConfig.model_validate(obj)

@pytest.mark.parametrize('key', ['total_updates', 'recovery_updates', 'temperature', 'recovery'])
def test_missing_contract_rejected(key):
    obj = policy_data()
    del obj[key]
    with pytest.raises(ValueError):
        module().PostQuantConfig.model_validate(obj)

@pytest.mark.parametrize('value', [-0.1, float('inf'), float('nan'), True, '1.0'])
def test_invalid_coefficients_rejected(value):
    obj = policy_data()
    obj['recovery']['option_kl'] = value
    with pytest.raises(ValueError):
        module().PostQuantConfig.model_validate(obj)

def test_unknown_nested_keys_and_wrong_direction_rejected():
    for part, key, value in [('recovery',
         'typo',
         1),
         ('respecialization_end',
         'option_kl',
         2.0),
         ('respecialization_end',
         'cross_entropy',
         0.1),
         ('respecialization_end',
         'brier',
         0.1)]:
        obj = policy_data()
        obj[part][key] = value
        with pytest.raises(ValueError):
            module().PostQuantConfig.model_validate(obj)

@pytest.mark.parametrize('index', [-1, 1024, True, 0.0, '0'])
def test_invalid_update_rejected(index):
    m = module()
    c = m.PostQuantConfig.model_validate(policy_data())
    with pytest.raises(ValueError):
        m.loss_weights_for_update(index, c)

def test_yaml_json_roundtrip_and_immutable_model(tmp_path):
    m = module()
    obj = policy_data()
    p = tmp_path / 'policy.yaml'
    p.write_text(json.dumps(obj))
    config = m.load_postquant_config(p)
    assert config.model_dump() == obj
    with pytest.raises(ValueError):
        config.total_updates = 8
    with pytest.raises(ValueError):
        config.recovery.option_kl = 7.0
    assert (m.policy_identity(config)
            == m.policy_identity(m.PostQuantConfig.model_validate(copy.deepcopy(obj))))
    obj['recovery']['option_kl'] = 1.1
    assert m.policy_identity(config) != m.policy_identity(m.PostQuantConfig.model_validate(obj))

@pytest.mark.parametrize('text',
     ['schema_version: 2\nschema_version: 2\n',
     '[]',
     '!!python/object:thing {}'])
def test_malformed_yaml_rejected(tmp_path, text):
    p = tmp_path / 'policy.yaml'
    p.write_text(text)
    with pytest.raises(ValueError):
        module().load_postquant_config(p)

def test_protected_path_not_opened_and_bounded_size(tmp_path):
    p = tmp_path / 'sealed' / 'policy.yaml'
    with pytest.raises(ValueError, match='protected'):
        module().load_postquant_config(p)
    p = tmp_path / 'policy.yaml'
    p.write_text('x' * 65537)
    with pytest.raises(ValueError, match='64 KiB'):
        module().load_postquant_config(p)

def test_example_config_matches_predeclared_values():
    m = module()
    p = Path(__file__).parents[1] / 'configs/recovery/decision_respecialization_v2.example.yaml'
    assert m.load_postquant_config(p).model_dump() == policy_data()
