from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def cache_module():
    name = 'tiny_omni_decision.teacher_cache'
    assert importlib.util.find_spec(name) is not None, 'teacher_cache implementation missing'
    return importlib.import_module(name)


def io_module():
    name = 'tiny_omni_decision.postquant_cache'
    assert importlib.util.find_spec(name) is not None, 'cache preparation implementation missing'
    return importlib.import_module(name)


def identity(n=2, sample_id='example-1', purpose='training'):
    labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')[:n]
    return {
        'purpose': purpose, 'sample_id': sample_id,
        'example_sha256': 'a' * 64, 'input_sha256': 'b' * 64,
        'option_texts': [f'option-{i}' for i in range(n)],
        'option_labels': labels, 'option_token_ids': list(range(100, 100 + n)),
        'target': n - 1, 'media_sha256s': [],
        'teacher': {'master_sha256': 'c' * 64, 'model_revision': 'd' * 40,
                    'processor_revision': 'e' * 40, 'preprocessing_sha256': 'f' * 64},
    }


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows),
                    encoding='utf-8')


@pytest.fixture
def job(tmp_path):
    frozen = tmp_path / 'frozen'
    frozen.mkdir()
    inputs = [identity(2, 'a'), identity(60, 'b')]
    exports = [{'identity': item, 'option_logits': [0.0] * len(item['option_texts'])}
               for item in inputs]
    write_rows(frozen / 'inputs.jsonl', inputs)
    write_rows(frozen / 'logits.jsonl', exports)
    manifest = {'schema_version': 1, 'kind': 'postquant_cache_import', 'purpose': 'training',
                'master_state': 'frozen_reload_verified', 'teacher': inputs[0]['teacher'],
                'expected_rows': 2,
                'inputs': {'path': 'frozen/inputs.jsonl',
                           'sha256': digest(frozen / 'inputs.jsonl')},
                'predictions': {'path': 'frozen/logits.jsonl',
                                'sha256': digest(frozen / 'logits.jsonl')}}
    path = tmp_path / 'job.json'
    write_json(path, manifest)
    return path, manifest, inputs, exports


def rewrite_source(job, role, rows):
    path, manifest, _, _ = job
    source = path.parent / manifest[role]['path']
    write_rows(source, rows)
    manifest[role]['sha256'] = digest(source)
    write_json(path, manifest)


@pytest.mark.parametrize('n', [2, 4, 10, 60, 62])
def test_record_roundtrip_and_exact_logits(n):
    m = cache_module()
    expected = identity(n)
    logits = [float(i) / 7 - 3 for i in range(n)]
    row = m.make_teacher_cache_record(expected, logits)
    m.validate_teacher_cache_record(row, expected)
    assert row['teacher_option_logits'] == logits
    assert row['identity'] == expected
    assert len(row['cache_key']) == 64
    assert row['temperature'] == 1.0
    expected['option_texts'][0] = 'mutated caller'
    logits[0] = 100.0
    assert row['identity']['option_texts'][0] == 'option-0'
    assert row['teacher_option_logits'][0] == -3.0


@pytest.mark.parametrize('field', ['purpose', 'sample_id', 'example_sha256', 'input_sha256',
                                   'option_texts', 'option_labels', 'option_token_ids', 'target',
                                   'media_sha256s', 'teacher'])
def test_stale_identity_rejected(field):
    m = cache_module()
    expected = identity()
    row = m.make_teacher_cache_record(expected, [1.0, 2.0])
    other = copy.deepcopy(expected)
    replacements = {'purpose': 'development', 'sample_id': 'different',
                    'example_sha256': '0' * 64, 'input_sha256': '0' * 64,
                    'option_texts': ['different', 'option-1'], 'option_labels': ['a', 'B'],
                    'option_token_ids': [400, 401], 'target': 0,
                    'media_sha256s': ['0' * 64],
                    'teacher': {**expected['teacher'], 'master_sha256': '0' * 64}}
    other[field] = replacements[field]
    with pytest.raises(ValueError):
        m.validate_teacher_cache_record(row, other)


@pytest.mark.parametrize('field,value', [
    ('target', True), ('target', -1), ('target', 2), ('option_token_ids', [1, 1]),
    ('option_token_ids', [True, 2]), ('option_token_ids', [-1, 2]),
    ('option_labels', ['A', 'A']), ('option_labels', ['a', 'b']),
    ('option_texts', ['same', 'same']), ('option_texts', ['', 'x']),
    ('purpose', 'test'), ('purpose', 'audit'), ('sample_id', ''),
    ('media_sha256s', ['bad']), ('example_sha256', 'bad'), ('input_sha256', 'BAD'),
])
def test_bad_identity_rejected(field, value):
    m = cache_module()
    item = identity()
    item[field] = value
    with pytest.raises(ValueError):
        m.make_teacher_cache_record(item, [1.0, 2.0])


@pytest.mark.parametrize('logits', [[0.0], [0.0, 1.0, 2.0], [True, 0.0],
                                   [float('nan'), 0.0], [float('inf'), 0.0], ['0', 1.0]])
def test_invalid_logits_rejected(logits):
    with pytest.raises(ValueError):
        cache_module().make_teacher_cache_record(identity(), logits)


def test_cache_key_and_unknown_fields_fail_closed():
    m = cache_module()
    row = m.make_teacher_cache_record(identity(), [1.0, 2.0])
    row['cache_key'] = '0' * 64
    with pytest.raises(ValueError):
        m.validate_teacher_cache_record(row, identity())
    row = m.make_teacher_cache_record(identity(), [1.0, 2.0])
    row['extra'] = True
    with pytest.raises(ValueError):
        m.validate_teacher_cache_record(row, identity())


def test_dry_run_reads_only_and_canonical_write(job):
    m = io_module()
    path, _, inputs, _ = job
    before = {str(p): p.read_bytes() for p in path.parent.rglob('*') if p.is_file()}
    output = path.parent / 'package'
    report = m.prepare_teacher_cache(path, output_dir=output)
    assert report['status'] == 'VALIDATED_NOT_WRITTEN'
    assert report['row_count'] == 2
    assert report['training_started'] is False
    assert not output.exists()
    assert before == {str(p): p.read_bytes() for p in path.parent.rglob('*') if p.is_file()}
    result = m.prepare_teacher_cache(path, output_dir=output, write=True)
    assert result['status'] == 'CACHE_PACKAGED_NOT_EXECUTED'
    assert result['cache_sha256'] == digest(output / 'cache.jsonl')
    manifest = json.loads((output / 'cache-manifest.json').read_text())
    assert (output / 'COMPLETE').read_text().strip() == digest(output / 'cache-manifest.json')
    assert manifest['row_count'] == 2
    rows = [json.loads(s) for s in (output / 'cache.jsonl').read_text().splitlines()]
    for row, expected in zip(rows, inputs, strict=True):
        cache_module().validate_teacher_cache_record(row, expected)
    assert result['cache_sha256'] == report['cache_sha256']


@pytest.mark.parametrize('change', ['missing', 'extra', 'duplicate', 'order', 'purpose',
                                    'teacher', 'token', 'legacy', 'nan'])
def test_bad_export_rejected_without_writes(job, change):
    m = io_module()
    path, _, _, exports = job
    if change == 'missing':
        exports.pop()
    elif change == 'extra':
        exports.append(copy.deepcopy(exports[0]))
    elif change == 'duplicate':
        exports[1] = copy.deepcopy(exports[0])
    elif change == 'order':
        exports.reverse()
    elif change == 'purpose':
        exports[0]['identity']['purpose'] = 'development'
    elif change == 'teacher':
        exports[0]['identity']['teacher']['master_sha256'] = '0' * 64
    elif change == 'token':
        exports[0]['identity']['option_token_ids'][0] = 77
    elif change == 'legacy':
        exports[0] = {'sample_id': 'a', 'option_logits': [0.0, 0.0]}
    else:
        exports[0]['option_logits'][0] = float('nan')
    rewrite_source(job, 'predictions', exports)
    with pytest.raises(ValueError):
        m.prepare_teacher_cache(path, output_dir=path.parent / 'package', write=True)
    assert not (path.parent / 'package').exists()


@pytest.mark.parametrize('field,value', [('schema_version', True), ('expected_rows', True),
                                        ('expected_rows', 3), ('purpose', 'audit'),
                                        ('master_state', 'training'), ('extra', 1)])
def test_bad_job_rejected(job, field, value):
    m = io_module()
    path, manifest, _, _ = job
    manifest[field] = value
    write_json(path, manifest)
    with pytest.raises(ValueError):
        m.prepare_teacher_cache(path)


def test_hash_mismatch_and_duplicate_json_key(job):
    m = io_module()
    path, manifest, _, _ = job
    manifest['predictions']['sha256'] = '0' * 64
    write_json(path, manifest)
    with pytest.raises(ValueError, match='hash'):
        m.prepare_teacher_cache(path)
    path.write_text('{"kind":"x", "kind":"y"}', encoding='utf-8')
    with pytest.raises(ValueError, match='duplicate'):
        m.prepare_teacher_cache(path)


@pytest.mark.parametrize('bad_path', ['../outside.jsonl', '/absolute.jsonl',
                                     r'C:\outside.jsonl', r'\\host\file',
                                     'https://example.org/file', 'frozen/sealed/rows.jsonl'])
def test_unsafe_path_rejected(job, bad_path):
    m = io_module()
    path, manifest, _, _ = job
    manifest['predictions']['path'] = bad_path
    write_json(path, manifest)
    with pytest.raises((ValueError, OSError)):
        m.prepare_teacher_cache(path)


def test_symlink_rejected(job):
    m = io_module()
    path, manifest, _, _ = job
    link = path.parent / 'linked.jsonl'
    try:
        link.symlink_to(path.parent / manifest['predictions']['path'])
    except OSError:
        pytest.skip('symlinks unavailable on this platform')
    manifest['predictions']['path'] = 'linked.jsonl'
    write_json(path, manifest)
    with pytest.raises(ValueError, match='symlink'):
        m.prepare_teacher_cache(path)


def test_existing_or_frozen_output_never_overwritten(job):
    m = io_module()
    path, _, _, _ = job
    output = path.parent / 'package'
    output.mkdir()
    (output / 'keep').write_text('keep')
    with pytest.raises((ValueError, OSError)):
        m.prepare_teacher_cache(path, output_dir=output, write=True)
    assert (output / 'keep').read_text() == 'keep'
    with pytest.raises(ValueError):
        m.prepare_teacher_cache(path, output_dir=path.parent / 'frozen/new', write=True)
    assert not (path.parent / 'frozen/new').exists()


def test_cli_default_no_ml_import_and_no_execute_flag(job):
    io_module()
    path, _, _, _ = job
    code = ('import sys; from tiny_omni_decision.postquant_cache import main; '
            'r=main(sys.argv[1:]); '
            'assert not any(x in sys.modules for x in ("torch", "transformers", "peft")); '
            'raise SystemExit(r)')
    result = subprocess.run([sys.executable, '-c', code, '--job', str(path)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['training_started'] is False
    result = subprocess.run([sys.executable, '-c', code, '--job', str(path), '--execute'],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 2


def test_package_verification_and_tamper_detection(job):
    m = io_module()
    assert hasattr(m, 'verify_teacher_cache_package'), 'package verification missing'
    path, _, _, _ = job
    output = path.parent / 'package'
    m.prepare_teacher_cache(path, output_dir=output, write=True)
    result = m.verify_teacher_cache_package(path, output)
    assert result['status'] == 'CACHE_VERIFIED_NOT_EXECUTED'
    original = (output / 'cache.jsonl').read_bytes()
    (output / 'cache.jsonl').write_bytes(original.replace(b'0.0', b'1.0', 1))
    with pytest.raises(ValueError, match='hash'):
        m.verify_teacher_cache_package(path, output)
    (output / 'cache.jsonl').write_bytes(original)
    (output / 'COMPLETE').unlink()
    with pytest.raises((ValueError, OSError)):
        m.verify_teacher_cache_package(path, output)


def test_recorded_master_identity_fields_are_all_binding():
    m = cache_module()
    expected = identity()
    row = m.make_teacher_cache_record(expected, [0.0, 0.0])
    for name, value in expected['teacher'].items():
        other = copy.deepcopy(expected)
        other['teacher'][name] = '0' * len(value)
        with pytest.raises(ValueError):
            m.validate_teacher_cache_record(row, other)


def test_preparation_interruption_never_marks_complete(job, monkeypatch):
    m = io_module()
    path, _, _, _ = job
    output = path.parent / 'package'
    original = m._scan

    def interrupted(manifest, sources, stream=None):
        if stream is not None:
            stream.write(b'partial')
            raise OSError('simulated disk failure')
        return original(manifest, sources, stream)

    monkeypatch.setattr(m, '_scan', interrupted)
    with pytest.raises(OSError, match='simulated'):
        m.prepare_teacher_cache(path, output_dir=output, write=True)
    assert not (output / 'COMPLETE').exists()
    assert not (output / 'cache-manifest.json').exists()
    assert (output / 'cache.jsonl').read_bytes() == b'partial'


def test_changed_source_between_scans_is_blocked(job, monkeypatch):
    m = io_module()
    path, manifest, _, _ = job
    output = path.parent / 'package'
    source = path.parent / manifest['predictions']['path']
    original = m._scan

    def changed(job_manifest, sources, stream=None):
        result = original(job_manifest, sources, stream)
        if stream is not None:
            source.write_bytes(source.read_bytes() + b' ')
        return result

    monkeypatch.setattr(m, '_scan', changed)
    with pytest.raises(ValueError, match='hash'):
        m.prepare_teacher_cache(path, output_dir=output, write=True)
    assert not (output / 'COMPLETE').exists()


def test_duplicate_inputs_rejected(job):
    m = io_module()
    path, _, inputs, exports = job
    inputs[1] = copy.deepcopy(inputs[0])
    exports[1] = copy.deepcopy(exports[0])
    rewrite_source(job, 'inputs', inputs)
    rewrite_source(job, 'predictions', exports)
    with pytest.raises(ValueError, match='duplicate'):
        m.prepare_teacher_cache(path)


def test_cli_verification_mode_and_conflicting_flags(job, capsys):
    m = io_module()
    assert hasattr(m, 'verify_teacher_cache_package'), 'package verification missing'
    path, _, _, _ = job
    output = path.parent / 'package'
    m.prepare_teacher_cache(path, output_dir=output, write=True)
    assert m.main(['--job', str(path), '--verify-package', str(output)]) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'CACHE_VERIFIED_NOT_EXECUTED'
    assert m.main(['--job', str(path), '--verify-package', str(output), '--write']) == 2
    assert json.loads(capsys.readouterr().err)['status'] == 'BLOCKED'


def test_standalone_script_help_has_no_training_switch():
    io_module()
    script = Path(__file__).resolve().parents[1] / 'scripts/prepare_postquant_cache.py'
    result = subprocess.run([sys.executable, str(script), '--help'], capture_output=True,
                            text=True, check=False)
    assert result.returncode == 0
    assert '--job' in result.stdout and '--write' in result.stdout
    assert '--execute' not in result.stdout
