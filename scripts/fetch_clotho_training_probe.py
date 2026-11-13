"""Acquire 16 deterministic exact-name training members; no corpus replacement."""

import hashlib
import html
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts.fetch_starss23_smoke import HttpRangeReader
from scripts.run_starss23_ipv4 import IPv4HTTPSHandler

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")


def main():
    source = ROOT / "clotho-aqa-development-split-01/train.json"
    data = source.read_bytes()
    expected = "4b3ee6fe483d20e55e32de47d491d946538df82ef7ce45698ec764d6aa0f1df3"
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("frozen train hash mismatch")
    if shutil.disk_usage(ROOT).free < 2 * 1024**3:
        raise ValueError("insufficient disk reserve")
    rows = {r["source_id"]: r for r in json.loads(data)}
    ordered = sorted(rows, key=lambda x: hashlib.sha256(f"17:{x}".encode()).hexdigest())
    output = ROOT / "clotho-aqa-training-probe-01"
    output.mkdir(exist_ok=False)
    report = {"status": "RUNNING", "selection": ordered[:16], "files": [],
              "train_sha256": expected, "not_a_replacement_corpus": True}
    reader = HttpRangeReader(
        "https://zenodo.org/api/records/6473207/files/audio_files.zip/content",
        3140677494, budget=64 * 1024**2, retries=2,
    )

    def save():
        report["transferred"] = reader.transferred
        (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    save()
    try:
        opener = urllib.request.build_opener(IPv4HTTPSHandler())
        with patch("urllib.request.urlopen", opener.open), zipfile.ZipFile(reader) as archive:
            lookup = {}
            for item in archive.infolist():
                if not item.filename.endswith(".wav"):
                    continue
                name = html.unescape(Path(item.filename).name)
                if name in lookup:
                    raise ValueError("ambiguous archive basename")
                lookup[name] = item
            selected = [(identity, lookup[html.unescape(rows[identity]["file_name"])])
                        for identity in report["selection"]]
            for identity, item in selected:
                if item.file_size > 16 * 1024**2:
                    raise ValueError("member too large")
                if rows[identity]["license"] not in {
                    "http://creativecommons.org/licenses/by/3.0/",
                    "http://creativecommons.org/publicdomain/zero/1.0/",
                }:
                    raise ValueError("unapproved license")
                content = archive.read(item)  # ZIP CRC enforced before saving.
                (output / f"{identity}.wav").write_bytes(content)
                report["files"].append({
                    "source_id": identity, "member": item.filename,
                    "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content),
                    "license": rows[identity]["license"], "uploader": rows[identity]["uploader"],
                })
                save()
                print(f"verified {len(report['files'])}/16", flush=True)
        report["status"] = "PASS"
    except Exception as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
