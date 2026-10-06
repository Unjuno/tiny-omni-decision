from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


@pytest.fixture
def launch():
    spec = importlib.util.find_spec('tiny_omni_decision.postquant_launch')
    assert spec is not None, 'The worker safety/launch module is not implemented'
    return importlib.import_module('tiny_omni_decision.postquant_launch')


@pytest.fixture
def case(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    git(repo, 'init', '-q', '-b', 'codex/worker')
    (repo / 'code.py').write_text('# tracked source\n')
    git(repo, 'add', '.')
    git(repo, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
        'commit', '-qm', 'fixture')
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    outputs = tmp_path / 'outputs'
    outputs.mkdir()
    roles = ['master', 'ternary', 'policy', 'train', 'development',
             'teacher_cache', 'initial_adapter']
    artifacts = {}
    for role in roles:
        path = inputs / f'{role}.json'
        path.write_text(json.dumps({'role': role}))
        artifacts[role] = {'path': path.name,
                           'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {
        'schema_version': 1, 'kind': 'postquant_launch',
        'arm': 'q3_recovery_to_respecialization', 'profile': 'smoke',
        'source_commit': git(repo, 'rev-parse', 'HEAD'), 'expected_branch': 'codex/worker',
        'input_root': str(inputs), 'output_root': str(outputs), 'run_name': 'smoke-q3',
        'artifacts': artifacts,
        'budget': {'total_updates': 8, 'recovery_updates': 2,
                   'gradient_accumulation_steps': 4},
        'limits': {'max_wall_seconds': 600, 'max_vram_mib': 14000,
                   'min_free_disk_mib': 1},
        'device': {'index': 0, 'name': 'Test GPU', 'min_total_vram_mib': 15000},
        'authorization': {'local_only': True, 'allow_paid_compute': False,
                          'allow_downloads': False, 'execute': False},
    }
    _write_policy(inputs, manifest)
    mpath = tmp_path / 'launch.json'
    def save():
        mpath.write_text(json.dumps(manifest), encoding='utf-8')
        return mpath
    save()
    return repo, inputs, outputs, manifest, save


def test_preflight_read_only_without_importing_torch(launch, case):
    repo, inputs, outputs, _, save = case
    before = {p: p.read_bytes() for p in inputs.iterdir()}
    had_torch = 'torch' in sys.modules
    report = launch.preflight(save(), repo)
    assert report['status'] == 'VALIDATED_NOT_EXECUTED'
    assert report['backend_available'] is False
    assert report['training_started'] is False
    assert list(outputs.iterdir()) == []
    assert before == {p: p.read_bytes() for p in inputs.iterdir()}
    assert ('torch' in sys.modules) == had_torch


@pytest.mark.parametrize('field,value', [
    ('schema_version', 2), ('schema_version', True), ('kind', 'train'),
    ('arm', 'all'), ('profile', 'full'), ('source_commit', 'HEAD'),
    ('expected_branch', 'main'), ('run_name', '../escape'), ('run_name', 'CON'),
])
def test_invalid_contract(launch, case, field, value):
    repo, _, _, manifest, save = case
    manifest[field] = value
    with pytest.raises(launch.LaunchError):
        launch.preflight(save(), repo)


@pytest.mark.parametrize('section,key,value', [
    ('authorization', 'allow_paid_compute', True),
    ('authorization', 'allow_downloads', True),
    ('authorization', 'local_only', False),
    ('authorization', 'execute', 'false'),
    ('budget', 'total_updates', 1024), ('budget', 'recovery_updates', 256),
    ('budget', 'gradient_accumulation_steps', True),
    ('limits', 'max_wall_seconds', 0), ('limits', 'max_vram_mib', -1),
    ('device', 'index', -1),
])
def test_invalid_nested_values(launch, case, section, key, value):
    repo, _, _, manifest, save = case
    manifest[section][key] = value
    with pytest.raises(launch.LaunchError):
        launch.preflight(save(), repo)


@pytest.mark.parametrize('path', ['../master.json', '/tmp/master.json',
    'C:\\sealed\\x', '\\\\server\\share\\x', 'audit/secret.json',
    'sealed-data/secret.json', 'final_audit/secret.json', 'x/../master.json'])
def test_forbidden_artifact_paths_rejected_before_read(launch, case, path):
    repo, _, _, manifest, save = case
    manifest['artifacts']['master']['path'] = path
    with pytest.raises(launch.LaunchError):
        launch.preflight(save(), repo)


def test_symlink_escape_rejected(launch, case, tmp_path):
    repo, inputs, _, manifest, save = case
    outside = tmp_path / 'secret.json'
    outside.write_text('outside')
    target = inputs / 'link.json'
    try:
        target.symlink_to(outside)
    except OSError:
        pytest.skip('symlinks unavailable on this host')
    manifest['artifacts']['master']['path'] = 'link.json'
    with pytest.raises(launch.LaunchError, match='symlink|escape'):
        launch.preflight(save(), repo)


def test_hash_mismatch(launch, case):
    repo, inputs, _, _, save = case
    (inputs / 'master.json').write_text('changed')
    with pytest.raises(launch.LaunchError, match='hash'):
        launch.preflight(save(), repo)


@pytest.mark.parametrize('change', ['dirty', 'head', 'branch', 'unknown', 'missing', 'exists'])
def test_source_and_destination_guards(launch, case, change):
    repo, _, outputs, manifest, save = case
    if change == 'dirty':
        (repo / 'code.py').write_text('changed')
    elif change == 'head':
        manifest['source_commit'] = 'a' * 40
    elif change == 'branch':
        manifest['expected_branch'] = 'codex/other'
    elif change == 'unknown':
        manifest['command'] = 'arbitrary shell command'
    elif change == 'missing':
        del manifest['artifacts']['teacher_cache']
    else:
        (outputs / manifest['run_name']).mkdir()
    with pytest.raises(launch.LaunchError):
        launch.preflight(save(), repo)


def test_duplicate_and_nonfinite_json(launch, tmp_path):
    path = tmp_path / 'launch.json'
    for value in ['{"kind": 1, "kind": 2}', '{"value": NaN}', '[]']:
        path.write_text(value)
        with pytest.raises(launch.LaunchError):
            launch.read_manifest(path)


@pytest.mark.parametrize('arm,updates,recovery', [
    ('q0_master', 0, 0), ('q1_ternary_raw', 0, 0),
    ('q2_fixed_recovery', 1024, 0),
    ('q3_recovery_to_respecialization', 1024, 256),
    ('q4_high_precision_control', 1024, 256),
])
def test_comparison_arm_budgets(launch, case, arm, updates, recovery):
    repo, inputs, _, manifest, save = case
    manifest.update(arm=arm, profile='comparison')
    manifest['budget'].update(total_updates=updates, recovery_updates=recovery)
    if updates:
        _write_policy(inputs, manifest)
    assert launch.preflight(save(), repo)['arm'] == arm


def test_execute_requires_both_opt_ins_and_real_backend(launch, case, monkeypatch):
    repo, _, outputs, manifest, save = case
    with pytest.raises(launch.LaunchError, match='authorization'):
        launch.dispatch(save(), repo, execute=True)
    manifest['authorization']['execute'] = True
    with pytest.raises(launch.LaunchError, match='backend'):
        launch.dispatch(save(), repo, execute=True)
    assert list(outputs.iterdir()) == []


def test_default_dispatch_does_not_load_backend(launch, case, monkeypatch):
    repo, _, outputs, _, save = case
    def fail(*args):
        pytest.fail('dry-run imported backend')
    monkeypatch.setattr(launch, 'load_backend', fail)
    assert launch.dispatch(save(), repo)['training_started'] is False
    assert list(outputs.iterdir()) == []


def test_resume_refused_until_backend_verified(launch, case):
    repo, _, _, _, save = case
    with pytest.raises(launch.LaunchError, match='resume'):
        launch.dispatch(save(), repo, resume_from=repo)


@pytest.mark.parametrize('script', ['postquant_preflight.py', 'run_postquant_smoke.py',
                                     'run_postquant_comparison.py'])
def test_script_help_and_strict_cli(script):
    result = subprocess.run([sys.executable, str(ROOT / 'scripts' / script), '--help'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--manifest' in result.stdout
    result = subprocess.run([sys.executable, str(ROOT / 'scripts' / script), '--exec'],
                            capture_output=True, text=True)
    assert result.returncode != 0


def test_smoke_cli_never_falls_back_to_longer_profile(launch, case):
    repo, _, outputs, manifest, save = case
    manifest['profile'] = 'comparison'
    manifest['budget'].update(total_updates=1024, recovery_updates=256)
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/run_postquant_smoke.py'),
                             '--manifest', str(save()), '--repo', str(repo)],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert 'profile' in result.stderr
    assert list(outputs.iterdir()) == []


def test_cli_valid_dryrun_and_gpu_probe_parsing(launch, case):
    repo, _, outputs, _, save = case
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/postquant_preflight.py'),
                            '--manifest', str(save()), '--repo', str(repo)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['training_started'] is False
    assert list(outputs.iterdir()) == []
    rows = launch.parse_gpu_csv('0, NVIDIA Test, 16384, 15000\n')
    assert rows[0]['total_vram_mib'] == 16384
    with pytest.raises(launch.LaunchError):
        launch.parse_gpu_csv('N/A, broken')


def test_tree_digest_is_deterministic_and_changes_with_contents(launch, tmp_path):
    root = tmp_path / 'checkpoint'
    root.mkdir()
    (root / 'z').write_text('weights')
    (root / 'a').write_text('config')
    digest = launch.artifact_hash(root)
    assert digest == launch.artifact_hash(root)
    (root / 'z').write_text('different')
    assert digest != launch.artifact_hash(root)


def test_tree_audit_rejected_before_any_file_hash(launch, tmp_path, monkeypatch):
    root = tmp_path / 'checkpoint'
    root.mkdir()
    (root / 'a').write_text('normal')
    (root / 'sealed').mkdir()
    (root / 'sealed/secret').write_text('never read')
    def fail(*args):
        pytest.fail('opened an artifact before rejecting sealed tree')
    monkeypatch.setattr(launch, '_file_hash', fail)
    with pytest.raises(launch.LaunchError, match='protected'):
        launch.artifact_hash(root)


def test_manifest_must_be_regular_file(launch, tmp_path, monkeypatch):
    if not hasattr(os, 'mkfifo'):
        pytest.skip('POSIX FIFO test')
    path = tmp_path / 'fifo'
    os.mkfifo(path)
    def fail(*args, **kwargs):
        pytest.fail('attempted to read non-regular manifest')
    monkeypatch.setattr(Path, 'read_text', fail)
    with pytest.raises(launch.LaunchError, match='regular'):
        launch.read_manifest(path)


def test_output_cannot_mutate_frozen_tree(launch, case):
    repo, inputs, _, manifest, save = case
    frozen = inputs / 'master-dir'
    frozen.mkdir()
    (frozen / 'weights').write_text('frozen')
    manifest['artifacts']['master'] = {'path': 'master-dir',
                                       'sha256': launch.artifact_hash(frozen)}
    manifest['output_root'] = str(frozen)
    with pytest.raises(launch.LaunchError, match='frozen|input'):
        launch.preflight(save(), repo)


def test_source_mutation_during_hashing_is_detected(launch, case, monkeypatch):
    repo, _, _, _, save = case
    original = launch.artifact_hash
    def changing(path):
        (repo / 'code.py').write_text('concurrent edit')
        return original(path)
    monkeypatch.setattr(launch, 'artifact_hash', changing)
    with pytest.raises(launch.LaunchError, match='source|dirty'):
        launch.preflight(save(), repo)


def test_handoff_is_one_arm_offline_and_restores_environment(launch, case, monkeypatch):
    from types import SimpleNamespace
    repo, _, outputs, manifest, save = case
    manifest['authorization']['execute'] = True
    seen = []
    def run(**kwargs):
        seen.append(kwargs)
        assert os.environ['HF_HUB_OFFLINE'] == '1'
        assert Path(str(kwargs['output_dir']) + '.launch.lock').is_file()
        assert not kwargs['output_dir'].exists()
        kwargs['output_dir'].mkdir()
        return {'fixture_only': True}
    monkeypatch.setattr(launch, 'load_backend', lambda root: SimpleNamespace(
        run_postquant_experiment=run))
    monkeypatch.setattr(launch, 'probe_device', lambda *args: {})
    monkeypatch.setattr(sys, 'version_info', (3, 11, 9))
    monkeypatch.setenv('HF_HUB_OFFLINE', 'original')
    result = launch.dispatch(save(), repo, execute=True)
    assert result['status'] == 'BACKEND_RETURNED'
    assert len(seen) == 1
    assert seen[0]['arm'] == manifest['arm']
    assert seen[0]['resume_from'] is None
    assert os.environ['HF_HUB_OFFLINE'] == 'original'
    assert not Path(str(outputs / manifest['run_name']) + '.launch.lock').exists()


def test_backend_failure_is_not_reported_as_success(launch, case, monkeypatch):
    from types import SimpleNamespace
    repo, _, outputs, manifest, save = case
    manifest['authorization']['execute'] = True
    def run(**kwargs):
        kwargs['output_dir'].mkdir()
        (kwargs['output_dir'] / 'failure.log').write_text('preserved')
        raise RuntimeError('fixture failure')
    monkeypatch.setattr(launch, 'load_backend', lambda root: SimpleNamespace(
        run_postquant_experiment=run))
    monkeypatch.setattr(launch, 'probe_device', lambda *args: {})
    monkeypatch.setattr(sys, 'version_info', (3, 11, 9))
    with pytest.raises(RuntimeError, match='fixture failure'):
        launch.dispatch(save(), repo, execute=True)
    assert (outputs / manifest['run_name'] / 'failure.log').read_text() == 'preserved'
    assert not Path(str(outputs / manifest['run_name']) + '.launch.lock').exists()


def test_gpu_identity_and_memory_checked(launch, monkeypatch):
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **k:
                        '0, Test GPU, 16384, 13000\n')
    with pytest.raises(launch.LaunchError, match='memory'):
        launch.probe_device({'index': 0, 'name': 'Test GPU', 'min_total_vram_mib': 15000},
                            {'max_vram_mib': 14000})
    with pytest.raises(launch.LaunchError, match='identity'):
        launch.probe_device({'index': 1, 'name': 'Test GPU', 'min_total_vram_mib': 15000},
                            {'max_vram_mib': 12000})


def test_offline_mode_covers_backend_import_and_failure(launch, case, monkeypatch):
    repo, _, outputs, manifest, save = case
    manifest['authorization']['execute'] = True
    monkeypatch.setenv('HF_HUB_OFFLINE', 'original')
    def loader(root):
        assert os.environ['HF_HUB_OFFLINE'] == '1'
        raise launch.LaunchError('backend unavailable')
    monkeypatch.setattr(launch, 'load_backend', loader)
    with pytest.raises(launch.LaunchError, match='backend unavailable'):
        launch.dispatch(save(), repo, execute=True)
    assert os.environ['HF_HUB_OFFLINE'] == 'original'
    assert list(outputs.iterdir()) == []


def _write_policy(inputs, manifest):
    fixed = manifest['arm'] == 'q2_fixed_recovery'
    policy = {
        'schema_version': 2, 'kind': 'postquant_policy',
        'arm': manifest['arm'], 'profile': manifest['profile'], **manifest['budget'],
        'temperature': 1.0, 'full_vocab_kl': 0.0,
        'schedule': 'fixed' if fixed else 'recovery_then_linear',
        'recovery': {'option_kl': 1.0, 'cross_entropy': 0.2, 'brier': 0.2},
        'respecialization_end': {'option_kl': 1.0 if fixed else 0.2,
                                'cross_entropy': 0.2 if fixed else 1.0, 'brier': 0.2},
    }
    path = inputs / 'policy.json'
    path.write_text(json.dumps(policy))
    manifest['artifacts']['policy']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


@pytest.mark.parametrize('key,value', [('schema_version', 1), ('arm', 'q4_high_precision_control'),
                                      ('total_updates', 1024), ('typo', 1)])
def test_preflight_checks_real_policy_not_only_hash(launch, case, key, value):
    repo, inputs, _, manifest, save = case
    path = inputs / 'policy.json'
    policy = json.loads(path.read_text())
    policy[key] = value
    path.write_text(json.dumps(policy))
    manifest['artifacts']['policy']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(launch.LaunchError, match='policy'):
        launch.preflight(save(), repo)


def test_preflight_reports_policy_identity_but_no_trainer(launch, case):
    repo, _, _, _, save = case
    report = launch.preflight(save(), repo)
    assert report['numerical_policy']['schema_version'] == 2
    assert len(report['numerical_policy']['identity']) == 64
    assert report['backend_available'] is False
