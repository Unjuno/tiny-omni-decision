from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
ROOT = Path(__file__).parents[1]
POLICY = ROOT / 'configs/recovery/decision_respecialization_v2.example.yaml'

def test_checker_exists():
    assert (ROOT / 'scripts/check_postquant_numerics.py').is_file()
    assert importlib.util.find_spec('tiny_omni_decision.postquant_check') is not None

def run(*args):
    return subprocess.run([sys.executable,
         str(ROOT / 'scripts/check_postquant_numerics.py'),
         *args],
         capture_output=True,
         text=True,
         timeout=30,
         env=os.environ | {'PYTHONPATH': str(ROOT / 'src'),
         'OMP_NUM_THREADS': '1'})

def test_default_reads_policy_without_torch_or_outputs(tmp_path):
    code = (
        'import builtins,sys; '
        'old=builtins.__import__; '
        "builtins.__import__=lambda name,*a,**k: "
        "(_ for _ in ()).throw(AssertionError('ML import')) "
        "if name=='torch' else old(name,*a,**k); "
        'from tiny_omni_decision.postquant_check import main; '
        "raise SystemExit(main(['--policy',sys.argv[1],'--completed-updates','256']))"
    )
    proc = subprocess.run([sys.executable,
         '-c',
         code,
         str(POLICY)],
         cwd=tmp_path,
         capture_output=True,
         text=True,
         timeout=30,
         env=os.environ | {'PYTHONPATH': str(ROOT / 'src')})
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report['status'] == 'POLICY_VALIDATED_NOT_EXECUTED'
    assert report['weights']['stage'] == 'respecialization'
    assert report['training_started'] is False and report['model_loaded'] is False
    assert list(tmp_path.iterdir()) == []

def test_numerical_selfcheck_uses_only_synthetic_cpu_tensors():
    pytest.importorskip('torch')
    proc = run('--policy', str(POLICY), '--check-numerics')
    assert proc.returncode == 0, proc.stderr
    r = json.loads(proc.stdout)
    assert r['training_started'] is False and r['gpu_used'] is False
    assert r['checks']['uniform_two_choice']['cross_entropy'] == pytest.approx(0.69314718)
    assert r['checks']['uniform_two_choice']['brier'] == 0.5
    assert r['checks']['teacher_gradient_absent'] is True
    assert r['checks']['quantizer_roundtrip'] is True
    assert r['checks']['reference_storage']['packed'] is False

@pytest.mark.parametrize('args',
     [('--execute',
    ),
     ('--pol',
     'anything'),
     ('--completed-updates',
     '1024')])
def test_no_execute_or_abbreviated_flags(args):
    assert run('--policy', str(POLICY), *args).returncode == 2
