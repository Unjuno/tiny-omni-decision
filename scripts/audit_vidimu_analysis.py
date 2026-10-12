"""Fetch checksum-pinned synchronization metadata; leave sensor payloads unopened."""

import hashlib
import json
import zipfile
from pathlib import Path

import requests

from scripts.range_generalization import sha, write


def main():
    output = Path("C:/CodexArtifacts/pretrained-reuse-r2/vidimu-analysis-audit-01")
    output.mkdir(exist_ok=False)
    source = Path("C:/CodexArtifacts/pretrained-reuse-r2/vidimu-source-audit-02")
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    entry = next(f for f in metadata["files"] if f["key"] == "analysis.zip")
    if (entry["size"], entry["checksum"]) != (
        169608328, "md5:6c937c7122fee77ff1ce306175cb26c7"
    ):
        raise ValueError("source pin mismatch")
    write(output / "source.json", entry)
    path = output / "analysis.zip"
    digest = hashlib.md5()
    received = 0
    with requests.get(entry["links"]["self"], stream=True, timeout=30) as response:
        response.raise_for_status()
        with path.open("xb") as target:
            for chunk in response.iter_content(1024 * 1024):
                received += len(chunk)
                if received > entry["size"]:
                    raise ValueError("size exceeded")
                digest.update(chunk)
                target.write(chunk)
    if received != entry["size"] or "md5:" + digest.hexdigest() != entry["checksum"]:
        raise ValueError("archive integrity failed")
    with zipfile.ZipFile(path) as archive:
        write(output / "inventory.json", [
            {"name": i.filename, "size": i.file_size, "crc": i.CRC}
            for i in archive.infolist()
        ])
        members = [i for i in archive.infolist()
                   if i.filename.rsplit("/", 1)[-1].lower() == "infotosync.csv"]
        if len(members) != 1 or members[0].file_size > 1_000_000:
            raise ValueError("unexpected synchronization metadata")
        sync = output / "infoToSync.csv"
        sync.write_bytes(archive.read(members[0]))
    write(output / "result.json", {
        "archive_sha256": sha(path), "sync_sha256": sha(sync),
        "member": members[0].filename, "sensor_payload_read": False,
    })
    print("Verified analysis archive and extracted synchronization metadata", flush=True)


if __name__ == "__main__":
    main()
