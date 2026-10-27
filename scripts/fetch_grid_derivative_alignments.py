"""Fetch a pinned derivative GRID alignment archive, without media."""

import argparse
import hashlib
import json
from pathlib import Path

import requests

URL = ('https://huggingface.co/datasets/SilentSpeak/EGCLLC/resolve/'
       'b61192d1a5790843425862111a0eb870d911e8a9/GRID_alignments.tar')
SHA = '40abb10300984775e6465b6ab6665231c8d81f3a9af559622d6f48a48d902f11'
SIZE = 67338240
REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    out = parser.parse_args().output_dir.resolve()
    if out.exists() or out.is_relative_to(REPO):
        raise ValueError('new external output required')
    out.mkdir(parents=True)
    sha, count = hashlib.sha256(), 0
    with requests.get(URL, stream=True, timeout=(20, 60)) as response:
        response.raise_for_status()
        with (out / 'alignments.tar.partial').open('xb') as file:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    count += len(chunk)
                    if count > SIZE:
                        raise ValueError('archive larger than pinned size')
                    sha.update(chunk)
                    file.write(chunk)
    if count != SIZE or sha.hexdigest() != SHA:
        raise ValueError('archive size or hash mismatch')
    (out / 'alignments.tar.partial').rename(out / 'alignments.tar')
    receipt = {'url': URL, 'sha256': SHA, 'bytes': count,
               'media_downloaded': False, 'license_claim': 'CC-BY-4.0 derivative card'}
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
