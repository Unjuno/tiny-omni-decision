"""Verify and decode the preselected original GRID s1 videos without fitting."""

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import av
import numpy as np

from scripts.audit_droid_train_temporal import digest

ROOT = Path('C:/CodexArtifacts/pretrained-reuse-r2')
ARCHIVE = ROOT / 'grid-video-s1-20261028-attempt-01/s1.zip'
MANIFEST = ROOT / 'grid-s1-smoke-manifest-20261028-attempt-01.json'
PINS = {ARCHIVE: '2eec254832a87bb7158e8d26ec0f8f1020b79739d09ef3ae325d294b51ca41f8',
        MANIFEST: '06155424e4a2c88a38afa5962c99af9f551a2a18cf8656c4c8e8125096e3a06b'}
REPO = Path(__file__).resolve().parents[1]


def validate_times(times):
    value = np.asarray(times, dtype=np.float64)
    if len(value) < 2 or not np.isfinite(value).all() or not (np.diff(value) > 0).all():
        raise ValueError('invalid video timestamps')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    if out.exists() or out.is_relative_to(REPO) or any(digest(p) != h for p, h in PINS.items()):
        raise ValueError('output or input hash mismatch')
    rows = json.loads(MANIFEST.read_bytes())['rows']
    if len(rows) != 16 or len({r['code'] for r in rows}) != 16:
        raise ValueError('unexpected smoke manifest')
    result_rows = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        if len(set(archive.namelist())) != len(archive.namelist()):
            raise ValueError('duplicate zip member')
        for row in rows:
            name = f"s1/{row['code']}.mpg"
            info = archive.getinfo(name)
            if info.file_size > 10_000_000:
                raise ValueError('video exceeds size bound')
            data = archive.read(info)  # zipfile validates member CRC
            with av.open(io.BytesIO(data)) as container:
                stream = container.streams.video[0]
                stream.codec_context.thread_count = 1
                times, dims = [], set()
                for frame in container.decode(stream):
                    if frame.pts is None or frame.time_base is None:
                        raise ValueError('video frame lacks timestamp')
                    times.append(float(frame.pts * frame.time_base))
                    dims.add((frame.width, frame.height))
                pts = validate_times(times)
                if len(dims) != 1:
                    raise ValueError('video dimensions change')
                selected = [int(np.abs(pts - t).argmin()) for t in np.linspace(pts[0], pts[-1], 16)]
                result_rows.append({'code': row['code'], 'sha256': hashlib.sha256(data).hexdigest(),
                                    'bytes': len(data), 'frames': len(pts),
                                    'dimensions': list(next(iter(dims))),
                                    'selected_indices': selected,
                                    'duration_seconds': float(pts[-1] - pts[0])})
    if any(digest(p) != h for p, h in PINS.items()):
        raise ValueError('input changed during verification')
    result = {'videos': result_rows, 'source_hashes': {str(k): v for k, v in PINS.items()},
              'av_version': av.__version__, 'script_sha256': digest(Path(__file__)),
              'model_run': False, 'quality_evaluation': False}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as file:
        json.dump(result, file, indent=2)
        file.write('\n')


if __name__ == '__main__':
    main()
