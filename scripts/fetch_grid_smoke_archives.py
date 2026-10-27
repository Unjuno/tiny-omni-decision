"""Fetch one fixed GRID smoke archive with bounded space and checksum checks."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import requests

SOURCES = {
    'audio': ('audio_25k.zip', '4b3ac37b1a258f55d1eebe657de491a9', 3_200_000_000),
    'video_s1': ('s1.zip', 'cbd6556668f061b5c3681bc722659b39', 600_000_000),
}
REPO = Path(__file__).resolve().parents[1]
RESERVE = 2 * 1024**3


def check_space(path, remaining):
    if shutil.disk_usage(path).free < RESERVE + remaining:
        raise ValueError('insufficient space for bounded archive and reserve')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, choices=tuple(SOURCES))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() or out.is_relative_to(REPO):
        raise ValueError('new external output directory required')
    filename, expected_md5, maximum = SOURCES[args.source]
    check_space(out.parent, maximum)
    out.mkdir(parents=True)
    url = f'https://zenodo.org/records/3625687/files/{filename}?download=1'
    md5, sha, count, next_report = hashlib.md5(), hashlib.sha256(), 0, 64 * 1024**2
    with requests.get(url, stream=True, timeout=(20, 60)) as response:
        response.raise_for_status()
        with (out / f'{filename}.partial').open('xb') as file:
            for chunk in response.iter_content(1024 * 1024):
                if not chunk:
                    continue
                count += len(chunk)
                if count > maximum:
                    raise ValueError('archive exceeds bounded size')
                md5.update(chunk)
                sha.update(chunk)
                file.write(chunk)
                if count >= next_report:
                    check_space(out, maximum - count)
                    print(f'{args.source}: {count} bytes', flush=True)
                    next_report += 64 * 1024**2
    if md5.hexdigest() != expected_md5:
        raise ValueError('publisher checksum mismatch; partial file retained')
    (out / f'{filename}.partial').rename(out / filename)
    result = {'url': url, 'source': args.source, 'bytes': count,
              'publisher_md5': expected_md5, 'sha256': sha.hexdigest(),
              'license': 'CC-BY-4.0, Zenodo GRID record 3625687'}
    (out / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
