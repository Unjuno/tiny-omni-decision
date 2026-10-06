"""Read-only worker preflight and fail-closed handoff; no ML training logic.

Launch-manifest v1 is separate from the planned numerical-policy schema v2.
The current repository has no postquant trainer: dry-run works; execution blocks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PureWindowsPath
from typing import Any

ARMS = ('q0_master', 'q1_ternary_raw', 'q2_fixed_recovery',
        'q3_recovery_to_respecialization', 'q4_high_precision_control')
ROLES = {'master', 'ternary', 'policy', 'train', 'development',
         'teacher_cache', 'initial_adapter'}
BACKEND = 'tiny_omni_decision.postquant_trainer'
PROTECTED = re.compile(r'(^|[-_.])(audit|sealed|heldout)([-_.]|$)', re.I)
HEX40 = re.compile(r'[0-9a-f]{40}')
HEX64 = re.compile(r'[0-9a-f]{64}')


class LaunchError(ValueError):
    """Invalid or unsupported execution request; do not start a run."""


def _keys(obj: Any, required: set[str], label: str) -> None:
    if not isinstance(obj, dict) or set(obj) != required:
        raise LaunchError(f'{label}: expected exactly these keys: {sorted(required)}')


def _integer(value: Any, minimum: int, label: str) -> None:
    if type(value) is not int or value < minimum:
        raise LaunchError(f'{label}: expected integer >= {minimum}')


def _protected(path: Path) -> None:
    if any(PROTECTED.search(part) for part in path.parts):
        raise LaunchError(f'protected audit/sealed path: {path}')


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise LaunchError(f'duplicate JSON key: {key}')
        obj[key] = value
    return obj


def _nonfinite(value: str) -> None:
    raise LaunchError(f'nonfinite JSON value: {value}')


def read_manifest(path: Path) -> dict[str, Any]:
    _protected(path)
    _protected(path.resolve())
    if not path.is_file():
        raise LaunchError('launch manifest must be a regular file')
    if path.stat().st_size > 1024 * 1024:
        raise LaunchError('launch manifest exceeds 1 MiB')
    try:
        obj = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_object,
                         parse_constant=_nonfinite)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise LaunchError(f'invalid launch JSON: {exc}') from exc
    if not isinstance(obj, dict):
        raise LaunchError('launch manifest must be a JSON object')
    return obj


def _root(value: Any, label: str) -> Path:
    if not isinstance(value, str) or value.startswith(('\\\\', '//')) or '://' in value:
        raise LaunchError(f'{label}: use an absolute local path, not a URL or UNC share')
    path = Path(value)
    if not path.is_absolute():
        raise LaunchError(f'{label}: absolute path for this operating system required')
    _protected(path)
    resolved = path.resolve(strict=True)
    _protected(resolved)
    if not resolved.is_dir():
        raise LaunchError(f'{label}: existing directory required')
    return resolved


def _inside(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value or ':' in value:
        raise LaunchError('artifact path must be a nonempty relative local path')
    windows = PureWindowsPath(value)
    path = Path(value.replace('\\', '/'))
    if path.is_absolute() or windows.drive or windows.root or '..' in path.parts:
        raise LaunchError(f'artifact path escapes input_root: {value}')
    _protected(path)
    cursor = root
    for part in path.parts:
        cursor /= part
        if cursor.is_symlink():
            raise LaunchError(f'symlink artifacts are unsupported: {cursor}')
    result = cursor.resolve(strict=True)
    if not result.is_relative_to(root):
        raise LaunchError(f'artifact path escapes input_root: {value}')
    _protected(result)
    return result


def _file_hash(path: Path) -> str:
    if not path.is_file():
        raise LaunchError(f'not a regular file: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        if not path.is_file():
            raise LaunchError(f'not a regular file: {path}')
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise LaunchError(f'artifact changed while hashing: {path}')
    return digest.hexdigest()


def artifact_hash(path: Path) -> str:
    """Files: SHA256(bytes). Trees: documented postquant-tree-v1 digest."""
    _protected(path)
    if path.is_symlink():
        raise LaunchError(f'symlink artifacts are unsupported: {path}')
    if path.is_file():
        return _file_hash(path)
    if not path.is_dir():
        raise LaunchError(f'not a file or directory: {path}')
    paths = sorted(path.rglob('*'), key=lambda p: p.relative_to(path).as_posix())
    # Inspect all paths before opening any content (including audit descendants).
    for child in paths:
        _protected(child)
        if child.is_symlink() or not (child.is_dir() or child.is_file()):
            raise LaunchError(f'unsupported tree entry: {child}')
    digest = hashlib.sha256(b'postquant-tree-v1\n')
    for child in paths:
        if child.is_file():
            row = [child.relative_to(path).as_posix(), _file_hash(child)]
            digest.update((json.dumps(row, ensure_ascii=True, separators=(',', ':'))
                           + '\n').encode('utf-8'))
    if not any(p.is_file() for p in paths):
        raise LaunchError('empty artifact tree')
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(['git', '-C', str(repo), *args],
                                       stderr=subprocess.PIPE, text=True, timeout=15).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise LaunchError(f'Git identity check failed: {exc}') from exc


def parse_gpu_csv(text: str) -> list[dict[str, Any]]:
    rows = []
    try:
        for row in csv.reader(text.strip().splitlines(), skipinitialspace=True):
            if len(row) != 4:
                raise ValueError('expected four GPU columns')
            rows.append({'index': int(row[0]), 'name': row[1].strip(),
                         'total_vram_mib': int(row[2]), 'free_vram_mib': int(row[3])})
    except ValueError as exc:
        raise LaunchError(f'invalid nvidia-smi output: {exc}') from exc
    return rows


def probe_device(device: dict[str, Any], limits: dict[str, Any]) -> dict[str, Any]:
    try:
        text = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free',
             '--format=csv,noheader,nounits'], text=True, stderr=subprocess.PIPE, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise LaunchError(f'GPU could not be checked: {exc}') from exc
    matches = [r for r in parse_gpu_csv(text) if r['index'] == device['index']]
    if len(matches) != 1 or matches[0]['name'] != device['name']:
        raise LaunchError('GPU identity does not match manifest')
    gpu = matches[0]
    if (gpu['total_vram_mib'] < device['min_total_vram_mib']
            or gpu['free_vram_mib'] < limits['max_vram_mib']):
        raise LaunchError('insufficient declared GPU memory headroom')
    return gpu


def preflight(manifest_path: Path, repo: Path, *, check_device: bool = False) -> dict[str, Any]:
    """Validate explicitly referenced artifacts only; never load models or datasets."""
    manifest = read_manifest(manifest_path)
    _keys(manifest, {'schema_version', 'kind', 'arm', 'profile', 'source_commit',
                    'expected_branch', 'input_root', 'output_root', 'run_name',
                    'artifacts', 'budget', 'limits', 'device', 'authorization'}, 'manifest')
    if type(manifest['schema_version']) is not int or manifest['schema_version'] != 1:
        raise LaunchError('unsupported launch schema_version')
    if manifest['kind'] != 'postquant_launch' or manifest['arm'] not in ARMS:
        raise LaunchError('invalid kind or arm; multi-arm execution is unsupported')
    profile, arm = manifest['profile'], manifest['arm']
    if profile not in ('smoke', 'comparison'):
        raise LaunchError('unknown profile')
    if profile == 'smoke' and arm in ARMS[:2]:
        raise LaunchError('smoke requires a trained arm (Q2/Q3/Q4)')
    _keys(manifest['authorization'], {'local_only', 'allow_paid_compute',
                                     'allow_downloads', 'execute'}, 'authorization')
    auth = manifest['authorization']
    if (any(type(v) is not bool for v in auth.values()) or not auth['local_only']
            or auth['allow_paid_compute'] or auth['allow_downloads']):
        raise LaunchError('authorization must prohibit paid compute and downloads')
    _keys(manifest['budget'], {'total_updates', 'recovery_updates',
                              'gradient_accumulation_steps'}, 'budget')
    total = (8 if profile == 'smoke' else 1024) if arm not in ARMS[:2] else 0
    stage = total // 4 if arm in ARMS[3:] else 0
    expected = {'total_updates': total, 'recovery_updates': stage,
                'gradient_accumulation_steps': 4}
    if manifest['budget'] != expected or any(type(v) is not int
                                            for v in manifest['budget'].values()):
        raise LaunchError(f'budget does not match arm/profile: expected {expected}')
    _keys(manifest['limits'], {'max_wall_seconds', 'max_vram_mib',
                              'min_free_disk_mib'}, 'limits')
    for key, value in manifest['limits'].items():
        _integer(value, 1, key)
    _keys(manifest['device'], {'index', 'name', 'min_total_vram_mib'}, 'device')
    _integer(manifest['device']['index'], 0, 'GPU index')
    _integer(manifest['device']['min_total_vram_mib'], 1, 'GPU capacity')
    if not isinstance(manifest['device']['name'], str) or not manifest['device']['name']:
        raise LaunchError('GPU name required')
    commit = manifest['source_commit']
    if not isinstance(commit, str) or HEX40.fullmatch(commit) is None:
        raise LaunchError('source_commit must be an immutable full Git SHA')
    branch = manifest['expected_branch']
    if not isinstance(branch, str) or not branch or branch in ('main', 'master'):
        raise LaunchError('explicit non-main implementation branch required')
    if _git(repo, 'rev-parse', 'HEAD') != commit:
        raise LaunchError('source_commit mismatch')
    if _git(repo, 'branch', '--show-current') != branch:
        raise LaunchError('branch mismatch')
    if _git(repo, 'status', '--porcelain', '--untracked-files=all'):
        raise LaunchError('dirty source checkout; commit or isolate changes first')
    input_root = _root(manifest['input_root'], 'input_root')
    output_root = _root(manifest['output_root'], 'output_root')
    name = manifest['run_name']
    if (not isinstance(name, str) or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name)
            is None or re.fullmatch(r'CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]', name, re.I)):
        raise LaunchError('run_name must be a portable single directory name')
    output = output_root / name
    _protected(output)
    if output.exists() or output.is_symlink() or Path(str(output) + '.launch.lock').exists():
        raise LaunchError('output or launch lock exists; never overwrite a run')
    if shutil.disk_usage(output_root).free < manifest['limits']['min_free_disk_mib'] * 1024**2:
        raise LaunchError('insufficient free disk space')
    artifacts = manifest['artifacts']
    if not isinstance(artifacts, dict) or not set(artifacts).issubset(ROLES):
        raise LaunchError('unknown artifact roles')
    required = {'master', 'policy', 'development'}
    if arm not in ARMS[:2]:
        required |= {'train', 'teacher_cache', 'initial_adapter'}
    if arm in ARMS[1:4]:
        required.add('ternary')
    if not required.issubset(artifacts):
        raise LaunchError(f'missing artifacts: {sorted(required - set(artifacts))}')
    paths = {}
    # Validate all names before hashing any artifacts.
    for role, record in artifacts.items():
        _keys(record, {'path', 'sha256'}, f'artifact {role}')
        if not isinstance(record['sha256'], str) or HEX64.fullmatch(record['sha256']) is None:
            raise LaunchError(f'{role}: invalid sha256')
        paths[role] = _inside(input_root, record['path'])
    for path in paths.values():
        if path.is_dir() and output.is_relative_to(path):
            raise LaunchError('output would mutate a frozen input tree')
    hashes = {}
    for role, path in paths.items():
        hashes[role] = artifact_hash(path)
        if hashes[role] != artifacts[role]['sha256']:
            raise LaunchError(f'{role}: artifact hash mismatch')
    if 'train' in hashes and hashes['train'] == hashes['development']:
        raise LaunchError('identical train/development artifacts')
    policy_info = None
    if arm not in ARMS[:2]:
        from .postquant_policy import load_postquant_config, policy_identity

        try:
            policy = load_postquant_config(paths['policy'])
        except (OSError, ValueError) as exc:
            raise LaunchError(f'numerical policy rejected: {exc}') from exc
        if (policy.arm != arm or policy.profile != profile
                or any(getattr(policy, key) != value for key, value in expected.items())):
            raise LaunchError('numerical policy does not match launch arm/profile/budget')
        if _file_hash(paths['policy']) != hashes['policy']:
            raise LaunchError('numerical policy changed during validation')
        policy_info = {'schema_version': 2, 'identity': policy_identity(policy)}
    if (_git(repo, 'rev-parse', 'HEAD') != commit
            or _git(repo, 'branch', '--show-current') != branch
            or _git(repo, 'status', '--porcelain', '--untracked-files=all')):
        raise LaunchError('source changed during preflight or is dirty')
    return {'status': 'VALIDATED_NOT_EXECUTED', 'training_started': False,
            'verification_scope': 'declared_files_only_not_deep_model_or_corpus_audit',
            'manifest_sha256': _file_hash(manifest_path), 'source_commit': commit,
            'arm': arm, 'profile': profile, 'budget': expected, 'output_dir': str(output),
            'artifact_hashes': hashes, 'numerical_policy': policy_info,
            'backend_available': (repo / 'src/tiny_omni_decision/postquant_trainer.py').is_file(),
            'gpu': probe_device(manifest['device'], manifest['limits']) if check_device else None,
            'python': sys.version.split()[0]}


def load_backend(repo: Path) -> Any:
    expected = repo.resolve() / 'src/tiny_omni_decision/postquant_trainer.py'
    if not expected.is_file():
        raise LaunchError('postquant backend is not implemented; no experiment started')
    module = importlib.import_module(BACKEND)
    capabilities = {'local_only', 'offline', 'resource_caps', 'exclusive_output', 'launch_v1'}
    if (Path(module.__file__).resolve() != expected
            or getattr(module, 'POSTQUANT_LAUNCH_API_VERSION', None) != 1
            or not capabilities.issubset(getattr(module, 'POSTQUANT_LAUNCH_CAPABILITIES', ()))
            or not callable(getattr(module, 'run_postquant_experiment', None))):
        raise LaunchError('backend does not implement the verified launch-v1 contract')
    return module


@contextmanager
def _offline_environment() -> Iterator[None]:
    offline = {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1'}
    previous = {key: os.environ.get(key) for key in offline}
    try:
        os.environ.update(offline)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def dispatch(manifest_path: Path, repo: Path, *, execute: bool = False,
             check_device: bool = False, resume_from: Path | None = None) -> dict[str, Any]:
    if resume_from is not None:
        raise LaunchError('resume is disabled until exact-resume integration is verified')
    report = preflight(manifest_path, repo, check_device=check_device)
    if not execute:
        return report
    manifest = read_manifest(manifest_path)
    if not manifest['authorization']['execute']:
        raise LaunchError('manifest authorization.execute must be true as well as --execute')
    # Offline flags cover imports too. They are not an OS network sandbox.
    with _offline_environment():
        backend = load_backend(repo)
        if not ((3, 11) <= sys.version_info[:2] < (3, 13)):
            raise LaunchError('execution requires the project Python 3.11/3.12 environment')
        probe_device(manifest['device'], manifest['limits'])
        # Backend owns run creation, exact state, caps and deep policy/data validation.
        if _file_hash(manifest_path) != report['manifest_sha256']:
            raise LaunchError('launch manifest changed during preflight')
        lock = Path(report['output_dir'] + '.launch.lock')
        try:
            with lock.open('x', encoding='utf-8') as stream:
                stream.write(report['manifest_sha256'] + '\n')
        except FileExistsError as exc:
            raise LaunchError('another launcher claimed this output') from exc
        try:
            result = backend.run_postquant_experiment(
                arm=manifest['arm'], run_manifest=manifest_path.resolve(),
                output_dir=Path(report['output_dir']), resume_from=None)
            return {'status': 'BACKEND_RETURNED', 'arm': manifest['arm'], 'result': result}
        finally:
            lock.unlink()


def main(mode: str, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Local postquant worker handoff (dry-run default)',
                                     allow_abbrev=False)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--check-device', action='store_true')
    if mode != 'preflight':
        parser.add_argument('--execute', action='store_true')
        parser.add_argument('--resume-from', type=Path)
    if mode == 'comparison':
        parser.add_argument('--arm', choices=ARMS, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = read_manifest(args.manifest)
        if mode != 'preflight' and manifest.get('profile') != mode:
            raise LaunchError(f'profile mismatch: this entrypoint requires {mode}')
        if mode == 'comparison' and manifest.get('arm') != args.arm:
            raise LaunchError('--arm does not match manifest; no overrides allowed')
        result = dispatch(args.manifest, args.repo, execute=getattr(args, 'execute', False),
                          check_device=args.check_device,
                          resume_from=getattr(args, 'resume_from', None))
        print(json.dumps(result, ensure_ascii=True, allow_nan=False, indent=2))
        return 0
    except (OSError, ImportError, TypeError, ValueError) as exc:
        print(json.dumps({'status': 'BLOCKED', 'error': str(exc)}, ensure_ascii=True),
              file=sys.stderr)
        return 2
