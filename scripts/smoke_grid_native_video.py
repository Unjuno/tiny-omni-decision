"""Run the pinned native VideoPrism worker on 16 fixed GRID smoke videos."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

from scripts.audit_droid_train_temporal import digest
from scripts.audit_grid_video_smoke import ARCHIVE, MANIFEST, PINS

ROOT = Path('C:/CodexArtifacts/pretrained-reuse-r2')
AUDIT = ROOT / 'grid-s1-video-smoke-20261028-attempt-01.json'
WORKER = Path('scripts/native_av_video_worker.py')
PINS = {**PINS, AUDIT: '5317e0841fbee8645298c7dc4c0466f5baf3b0925a15e840866c113d94f9f8d1',
        WORKER: '5857a1d8973c7995c67efed448a87561e43a64ed122e49446d1d1e3f1a074093'}
REPO = Path(__file__).resolve().parents[1]


def validate_feature(value):
    feature = np.asarray(value, dtype=np.float32)
    if feature.shape != (16, 768) or not np.isfinite(feature).all():
        raise ValueError('invalid native video feature')
    return feature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    out = parser.parse_args().output_dir.resolve()
    if out.exists() or out.is_relative_to(REPO) or any(digest(p) != h for p, h in PINS.items()):
        raise ValueError('output or pinned input mismatch')
    rows = json.loads(MANIFEST.read_bytes())['rows']
    audit = json.loads(AUDIT.read_bytes())['videos']
    if len(rows) != 16 or [r['code'] for r in rows] != [r['code'] for r in audit]:
        raise ValueError('manifest and video audit mismatch')
    out.mkdir(parents=True)
    media = out / 'media'
    media.mkdir()
    paths = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for row, check in zip(rows, audit, strict=True):
            data = archive.read(f"s1/{row['code']}.mpg")
            if hashlib.sha256(data).hexdigest() != check['sha256']:
                raise ValueError('member hash mismatch')
            path = media / (check['sha256'] + '.mpg')
            path.write_bytes(data)
            paths.append((path, check['sha256']))
    deps = Path('C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-r1/venv/Lib/site-packages')
    source = Path('C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-r1/source')
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(map(str, (deps, source, REPO, REPO / 'src')))
    env['JAX_PLATFORMS'] = 'cpu'
    env['TF_CPP_MIN_LOG_LEVEL'] = '2'
    command = [sys.executable, '-m', 'scripts.native_av_video_worker',
               '--output', str(out / 'worker')]
    started = time.perf_counter()
    with (out / 'worker.log').open('w', encoding='utf-8') as log:
        worker = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=log, text=True, encoding='utf-8',
                                  creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            ready = json.loads(worker.stdout.readline())
            if not ready.get('ready'):
                raise ValueError('worker failed to initialize')
            calls = []
            for path, sha in paths:
                worker.stdin.write(json.dumps({'path': str(path), 'sha256': sha}) + '\n')
                worker.stdin.flush()
                reply = json.loads(worker.stdout.readline())
                feature = validate_feature(reply['feature'])
                calls.append({'media_sha256': sha,
                              'feature_sha256': hashlib.sha256(feature.tobytes()).hexdigest(),
                              'metadata': reply['metadata']})
                print(f'encoded {len(calls)} / 16', flush=True)
            worker.stdin.close()
            if worker.wait(timeout=120) != 0:
                raise RuntimeError('worker exited unsuccessfully')
        finally:
            if worker.poll() is None:
                worker.terminate()
                worker.wait(timeout=10)
    terminal = json.loads((out / 'worker/result.json').read_bytes())
    if terminal['status'] != 'PASS_NATIVE_VIDEO_WORKER' or len(calls) != 16:
        raise ValueError('worker did not complete integrity checks')
    if any(digest(p) != h for p, h in PINS.items()):
        raise ValueError('input changed during run')
    result = {'calls': calls, 'seconds': time.perf_counter() - started,
              'worker_result_sha256': digest(out / 'worker/result.json'),
              'source_hashes': {str(k): v for k, v in PINS.items()},
              'script_sha256': digest(Path(__file__)), 'gpu_used': False,
              'training': False, 'quality_evaluation': False}
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
