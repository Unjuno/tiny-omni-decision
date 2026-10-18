"""Fetch the fixed856-source development corpus with four bounded range readers."""

import argparse
import hashlib
import html
import json
import shutil
import threading
import time
import urllib.error
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


class TransientRangeReader(HttpRangeReader):
    """Retry explicit transient gateway responses without accepting invalid ranges."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transient_errors = []

    def read(self, size=-1):
        for attempt in range(3):
            try:
                return super().read(size)
            except urllib.error.HTTPError as error:
                self.transient_errors.append({"status": error.code, "position": self.position})
                if error.code not in {502, 503, 504} or attempt == 2:
                    raise
                time.sleep(attempt + 1)


def reusable_records(source, expected):
    report = json.loads((source / "result.json").read_text())
    if report["status"] not in {"PASS", "FAIL"} or report["split_hashes"] != HASHES:
        raise ValueError("reuse requires terminal acquisition with identical splits")
    reuse = {}
    for row in report["files"]:
        identity = row["source_id"]
        if identity not in expected or identity in reuse:
            raise ValueError("unexpected or duplicate reuse identity")
        wanted = expected[identity]
        if any(row[k] != wanted[k] for k in ("license", "uploader", "split")) or html.unescape(
            row["member"]
        ) != "audio_files/" + media_basename(wanted):
            raise ValueError("reuse provenance mismatch")
        path = source / (identity + ".wav")
        content = path.read_bytes()
        if len(content) != row["bytes"] or hashlib.sha256(content).hexdigest() != row["sha256"]:
            raise ValueError("reuse media hash/length mismatch")
        reuse[identity] = (path, row["sha256"])
    return reuse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "clotho-aqa-development-media-02")
    parser.add_argument("--reuse", type=Path)
    args = parser.parse_args()
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
    if args.reuse is not None:
        reuse.update(reusable_records(args.reuse, rows))
    output = args.output
    output.mkdir(exist_ok=False)
    lock = threading.Lock()
    readers = [
        TransientRangeReader(
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
        reused_acquisition=str(args.reuse) if args.reuse else None,
    )

    def save():
        report["transferred"] = sum(r.transferred for r in readers)
        report["transient_http_errors"] = [error for r in readers for error in r.transient_errors]
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
