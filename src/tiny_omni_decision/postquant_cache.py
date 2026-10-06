"""Validate and package already-exported teacher logits. Never runs inference/training."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import shutil
import stat
import sys
from pathlib import Path, PureWindowsPath
from typing import Any, BinaryIO

from .teacher_cache import (
    canonical_json,
    exact_keys,
    make_teacher_cache_record,
    require_digest,
    validate_cache_identity,
    validate_teacher_cache_record,
    validate_teacher_identity,
)

MAX_LINE = 1024 * 1024
PROTECTED = re.compile(r'(^|[-_.])(audit|sealed|heldout)([-_.]|$)', re.I)
JOB_KEYS = {'schema_version', 'kind', 'purpose', 'master_state', 'teacher', 'expected_rows',
            'inputs', 'predictions'}


def _local(path: Path, *, file: bool = True) -> Path:
    path = path.absolute()
    for item in (path, *path.parents):
        if PROTECTED.search(item.name):
            raise ValueError(f'protected audit/sealed path: {item}')
        if item.is_symlink():
            raise ValueError(f'symlink path is unsupported: {item}')
        if item.exists() and getattr(item.stat(), 'st_file_attributes', 0) & 0x400:
            raise ValueError(f'Windows reparse path is unsupported: {item}')
    resolved = path.resolve(strict=file)
    if file and not stat.S_ISREG(resolved.stat().st_mode):
        raise ValueError('regular local file required')
    return resolved


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError(f'nonfinite JSON literal: {value}')


def _decode(raw: bytes) -> dict[str, Any]:
    if len(raw) > MAX_LINE:
        raise ValueError('JSON object exceeds 1 MiB')
    try:
        data = json.loads(raw.decode('utf-8'), object_pairs_hook=_pairs,
                          parse_constant=_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'invalid UTF-8 JSON: {exc}') from exc
    if not isinstance(data, dict):
        raise ValueError('JSON object required')
    return data


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        for chunk in iter(lambda: stream.read(MAX_LINE), b''):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'file changed while hashing: {path}')
    return digest.hexdigest()


def _source(job_path: Path, descriptor: dict[str, Any]) -> Path:
    exact_keys(descriptor, {'path', 'sha256'}, 'source descriptor')
    require_digest(descriptor['sha256'])
    value = descriptor['path']
    if not isinstance(value, str) or not value or ':' in value:
        raise ValueError('use a local relative source path')
    p = Path(value.replace('\\', '/'))
    win = PureWindowsPath(value)
    if p.is_absolute() or win.drive or win.root or '..' in p.parts:
        raise ValueError('source path escapes the job directory')
    return _local(job_path.parent / p)


def _rows(path: Path):
    with path.open('rb') as stream:
        while raw := stream.readline(MAX_LINE + 1):
            if not raw.strip():
                raise ValueError(f'blank JSONL row: {path}')
            yield _decode(raw)


def _scan(job: dict[str, Any], sources: dict[str, Path],
          stream: BinaryIO | None = None) -> dict[str, Any]:
    cache_hash, order_hash = hashlib.sha256(), hashlib.sha256()
    seen: set[str] = set()
    count, byte_count = 0, 0
    rows = itertools.zip_longest(_rows(sources['inputs']), _rows(sources['predictions']))
    for expected, prediction in rows:
        if expected is None or prediction is None:
            raise ValueError('inputs and predictions have different row counts')
        count += 1
        if count > job['expected_rows']:
            raise ValueError('more rows than the declared export')
        validate_cache_identity(expected)
        if expected['purpose'] != job['purpose'] or expected['teacher'] != job['teacher']:
            raise ValueError('input purpose or teacher differs from the declared job')
        if expected['sample_id'] in seen:
            raise ValueError('duplicate sample_id; export each exact input once')
        seen.add(expected['sample_id'])
        exact_keys(prediction, {'identity', 'option_logits'}, 'raw export row')
        row = make_teacher_cache_record(prediction['identity'], prediction['option_logits'])
        validate_teacher_cache_record(row, expected)
        serialized = canonical_json(row) + b'\n'
        cache_hash.update(serialized)
        order_hash.update(canonical_json(expected['sample_id']) + b'\n')
        byte_count += len(serialized)
        if stream is not None:
            stream.write(serialized)
    if count != job['expected_rows']:
        raise ValueError('export row count does not match expected_rows')
    return {'row_count': count, 'cache_sha256': cache_hash.hexdigest(),
            'ordered_ids_sha256': order_hash.hexdigest(), 'cache_bytes': byte_count}


def _unchanged(paths: dict[str, Path], hashes: dict[str, str]) -> None:
    for role, path in paths.items():
        if _local(path) != path or _hash(path) != hashes[role]:
            raise ValueError(f'{role}: source hash mismatch or source changed')


def prepare_teacher_cache(job_path: Path, *, output_dir: Path | None = None,
                          write: bool = False) -> dict[str, Any]:
    """Check all rows first; writing requires an explicit flag and a new directory.

    master_state is a producer assertion, not a reload verification performed here.
    Legacy validation rows without captured input/token identity are rejected.
    """
    job_path = _local(Path(job_path))
    with job_path.open('rb') as stream:
        raw = stream.read(MAX_LINE + 1)
    job = _decode(raw)
    exact_keys(job, JOB_KEYS, 'cache import job')
    if type(job['schema_version']) is not int or job['schema_version'] != 1:
        raise ValueError('unsupported import schema')
    if job['kind'] != 'postquant_cache_import' or job['purpose'] not in ('training', 'development'):
        raise ValueError('require a training/development cache import, not audit/test')
    if job['master_state'] != 'frozen_reload_verified':
        raise ValueError('export producer must have a frozen, reload-verified Master')
    if type(job['expected_rows']) is not int or not 1 <= job['expected_rows'] <= 1_000_000:
        raise ValueError('expected_rows must be an integer from 1 to 1000000')
    validate_teacher_identity(job['teacher'])
    # Resolve every name before reading either source. No directory scan, media or model read.
    sources = {role: _source(job_path, job[role]) for role in ('inputs', 'predictions')}
    if len({job_path, *sources.values()}) != 3:
        raise ValueError('job, inputs and predictions must be distinct files')
    paths = {'job': job_path, **sources}
    hashes = {'job': hashlib.sha256(raw).hexdigest(),
              **{role: job[role]['sha256'] for role in sources}}
    output = _local(Path(output_dir), file=False) if output_dir is not None else None
    if write and output is None:
        raise ValueError('--write requires an explicit new output directory')
    if output is not None:
        if output.exists():
            raise ValueError('output exists; never overwrite an existing package')
        if not output.parent.is_dir():
            raise ValueError('output parent must already exist')
        if any(output.is_relative_to(p.parent) for p in sources.values()):
            raise ValueError('output must not be inside the frozen source directories')
    _unchanged(paths, hashes)
    summary = _scan(job, sources)
    _unchanged(paths, hashes)
    report = {'status': 'VALIDATED_NOT_WRITTEN', 'training_started': False,
              'model_loaded': False, 'verification_scope': 'export_identity_not_model_reexecution',
              'purpose': job['purpose'], 'teacher': job['teacher'], **summary,
              'source_hashes': hashes, 'output_dir': str(output) if output is not None else None}
    if not write:
        return report
    if shutil.disk_usage(output.parent).free < summary['cache_bytes'] + MAX_LINE:
        raise ValueError('insufficient free disk for the verified cache package')
    output.mkdir(exist_ok=False)
    # On interruption retain the partial directory WITHOUT COMPLETE. Never overwrite/reuse it.
    with (output / 'cache.jsonl').open('xb') as stream:
        repeated = _scan(job, sources, stream)
        stream.flush()
        os.fsync(stream.fileno())
    if repeated != summary:
        raise ValueError('source changed between validation and publication')
    _unchanged(paths, hashes)
    if _hash(output / 'cache.jsonl') != summary['cache_sha256']:
        raise ValueError('written cache hash mismatch')
    manifest = {'schema_version': 1, 'kind': 'postquant_teacher_cache',
                'purpose': job['purpose'], 'teacher': job['teacher'], **summary,
                'source_hashes': hashes}
    manifest_bytes = canonical_json(manifest) + b'\n'
    for name, content in (
        ('cache-manifest.json', manifest_bytes),
        ('COMPLETE', (hashlib.sha256(manifest_bytes).hexdigest() + '\n').encode('ascii')),
    ):
        with (output / name).open('xb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    return {**report, 'status': 'CACHE_PACKAGED_NOT_EXECUTED'}



def verify_teacher_cache_package(job_path: Path, package_dir: Path) -> dict[str, Any]:
    """Revalidate immutable sources, package hashes and the final completion marker."""
    package = _local(Path(package_dir), file=False)
    if not package.is_dir() or {p.name for p in package.iterdir()} != {
        'cache.jsonl', 'cache-manifest.json', 'COMPLETE'
    }:
        raise ValueError('cache package incomplete or contains unexpected files')
    files = {name: _local(package / name)
             for name in ('cache.jsonl', 'cache-manifest.json', 'COMPLETE')}
    expected = prepare_teacher_cache(job_path)
    with files['cache-manifest.json'].open('rb') as stream:
        manifest_bytes = stream.read(MAX_LINE + 1)
    manifest = _decode(manifest_bytes)
    wanted = {'schema_version': 1, 'kind': 'postquant_teacher_cache',
              **{key: expected[key] for key in ('purpose', 'teacher', 'row_count',
                 'cache_sha256', 'ordered_ids_sha256', 'cache_bytes', 'source_hashes')}}
    if canonical_json(manifest) != canonical_json(wanted):
        raise ValueError('cache manifest does not match the frozen export job')
    with files['COMPLETE'].open('rb') as stream:
        complete = stream.read(66)
    if complete != (hashlib.sha256(manifest_bytes).hexdigest() + '\n').encode('ascii'):
        raise ValueError('cache completion marker hash mismatch')
    if _hash(files['cache.jsonl']) != expected['cache_sha256']:
        raise ValueError('cache content hash mismatch')
    return {**expected, 'status': 'CACHE_VERIFIED_NOT_EXECUTED', 'output_dir': str(package)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify-package', type=Path)
    parser.add_argument('--write', action='store_true', help='Package already-exported logits only')
    args = parser.parse_args(argv)
    try:
        if args.verify_package is not None:
            if args.write or args.output is not None:
                raise ValueError('--verify-package cannot be combined with --write/--output')
            report = verify_teacher_cache_package(args.job, args.verify_package)
        else:
            report = prepare_teacher_cache(args.job, output_dir=args.output, write=args.write)
        print(json.dumps(report, ensure_ascii=True, allow_nan=False, indent=2))
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({'status': 'BLOCKED', 'training_started': False, 'error': str(exc)}),
              file=sys.stderr)
        return 2
