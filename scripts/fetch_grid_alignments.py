"""Fetch the public GRID alignment archive only, validating the publisher checksum."""

import argparse
import hashlib
import json
from pathlib import Path

import requests

URL = 'https://zenodo.org/records/3625687/files/alignments.zip?download=1'
EXPECTED_MD5 = 'df892a5adea6595339b331cc065501f9'
REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    out = parser.parse_args().output_dir.resolve()
    if out.exists() or out.is_relative_to(REPO):
        raise ValueError('new external output directory required')
    out.mkdir(parents=True)
    md5, sha, size = hashlib.md5(), hashlib.sha256(), 0  # noqa: S324
    with requests.get(URL, stream=True, timeout=(20, 60)) as response:
        response.raise_for_status()
        with (out / 'alignments.zip.partial').open('xb') as stream:
            for chunk in response.iter_content(1024 * 1024):
                if not chunk:
                    continue
                size += len(chunk)
                if size > 30_000_000:
                    raise ValueError('archive exceeds bounded size')
                md5.update(chunk)
                sha.update(chunk)
                stream.write(chunk)
    if md5.hexdigest() != EXPECTED_MD5:
        raise ValueError('publisher checksum mismatch; preserve partial artifact')
    (out / 'alignments.zip.partial').rename(out / 'alignments.zip')
    receipt = {'source': URL, 'publisher_md5': EXPECTED_MD5, 'sha256': sha.hexdigest(),
               'bytes': size, 'media_downloaded': False,
               'data_license': 'CC-BY-4.0 as stated by Zenodo record 3625687'}
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
