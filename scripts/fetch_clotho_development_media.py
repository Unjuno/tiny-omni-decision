"""Fetch the fixed856-source development corpus with four bounded range readers."""

import hashlib
import html
import json
import shutil
import threading
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from scripts.clotho_media_identity import ALIAS_SHA256, media_basename
from scripts.fetch_starss23_smoke import HttpRangeReader
from scripts.run_starss23_ipv4 import IPv4HTTPSHandler

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
HASHES = {
    "train": "4b3ee6fe483d20e55e32de47d491d946538df82ef7ce45698ec764d6aa0f1df3",
    "calibration": "dcdafb80a88999a5e1c401883d8b0a6c520a2e50059b1364769336e500c74291",
    "validation": "9587abbc775e03698f48bd10d45ac8fd8703018689999d3ed8925f7d2bf0527f",
}


def main():
    rows = {}
    for split, expected in HASHES.items():
        data = (ROOT / f"clotho-aqa-development-split-01/{split}.json").read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("frozen split changed")
        for row in json.loads(data):
            identity = row["source_id"]
            if identity in rows and rows[identity]["split"] != split:
                raise ValueError("source split overlap")
            if row["license"] not in {
                "http://creativecommons.org/licenses/by/3.0/",
                "http://creativecommons.org/publicdomain/zero/1.0/",
            }:
                raise ValueError("unapproved license")
            rows[identity] = dict(row, split=split)
    if len(rows) != 856 or shutil.disk_usage(ROOT).free < 5 * 1024**3:
        raise ValueError("unexpected corpus or insufficient disk")
    old = ROOT / "clotho-aqa-training-probe-01"
    old_report = json.loads((old / "result.json").read_text())
    if old_report["status"] != "PASS":
        raise ValueError("reuse source incomplete")
    reuse = {
        r["source_id"]: (old / (r["source_id"] + ".wav"), r["sha256"]) for r in old_report["files"]
    }
    reuse["16854"] = (ROOT / "clotho-aqa-alias-waveform-01/candidate.wav", ALIAS_SHA256)
    output = ROOT / "clotho-aqa-development-media-01"
    output.mkdir(exist_ok=False)
    lock = threading.Lock()
    readers = [
        HttpRangeReader(
            "https://zenodo.org/api/records/6473207/files/audio_files.zip/content",
            3140677494,
            budget=512 * 1024**2,
            retries=2,
        )
        for _ in range(4)
    ]
    report = dict(
        status="RUNNING",
        split_hashes=HASHES,
        files=[],
        workers=4,
        started=time.time(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )

    def save():
        report["transferred"] = sum(r.transferred for r in readers)
        temporary = output / "result.tmp"
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(output / "result.json")

    def worker(number, identities):
        with zipfile.ZipFile(readers[number]) as archive:
            lookup = {}
            for item in archive.infolist():
                if item.filename.endswith(".wav"):
                    key = html.unescape(Path(item.filename).name)
                    if key in lookup:
                        raise ValueError("duplicate archive basename")
                    lookup[key] = item
            for identity in identities:
                row = rows[identity]
                item = lookup[media_basename(row)]
                if item.file_size > 16 * 1024**2:
                    raise ValueError("member size exceeds bound")
                if identity in reuse:
                    path, expected = reuse[identity]
                    content = path.read_bytes()
                    if hashlib.sha256(content).hexdigest() != expected:
                        raise ValueError("reuse hash mismatch")
                else:
                    content = archive.read(item)
                if len(content) != item.file_size or zlib.crc32(content) != item.CRC:
                    raise ValueError("member length/CRC mismatch")
                digest = hashlib.sha256(content).hexdigest()
                if identity == "16854" and digest != ALIAS_SHA256:
                    raise ValueError("reviewed alias content changed")
                if shutil.disk_usage(output).free < len(content) + 2 * 1024**3:
                    raise ValueError("disk reserve exhausted")
                (output / (identity + ".wav")).write_bytes(content)
                with lock:
                    report["files"].append(
                        dict(
                            source_id=identity,
                            split=row["split"],
                            member=item.filename,
                            sha256=digest,
                            bytes=len(content),
                            license=row["license"],
                            uploader=row["uploader"],
                            reused=identity in reuse,
                        )
                    )
                    save()
                    if len(report["files"]) % 16 == 0:
                        print(f"verified {len(report['files'])}/856", flush=True)

    save()
    try:
        ordered = sorted(rows)
        opener = urllib.request.build_opener(IPv4HTTPSHandler())
        with patch("urllib.request.urlopen", opener.open), ThreadPoolExecutor(4) as pool:
            futures = [pool.submit(worker, i, ordered[i::4]) for i in range(4)]
            for future in futures:
                future.result()
        if len(report["files"]) != 856:
            raise ValueError("incomplete acquisition")
        report["status"] = "PASS"
    except Exception as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        report["ended"] = time.time()
        save()


if __name__ == "__main__":
    main()
