"""Fetch checksum-pinned synchronization metadata; leave sensor payloads unopened."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from scripts.range_generalization import sha, write


def verify_archive(path, *, size, checksum):
    if path.stat().st_size != size:
        raise ValueError("archive size mismatch")
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if "md5:" + digest.hexdigest() != checksum:
        raise ValueError("archive checksum mismatch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-existing", type=Path)
    args = parser.parse_args()
    output = args.verify_existing or Path(
        "C:/CodexArtifacts/pretrained-reuse-r2/vidimu-analysis-audit-01"
    )
    if not args.verify_existing:
        output.mkdir(exist_ok=False)
    elif any(
        (output / name).exists() for name in ("result.json", "inventory.json", "infoToSync.csv")
    ):
        raise ValueError("refusing to overwrite existing inspection")
    source = Path("C:/CodexArtifacts/pretrained-reuse-r2/vidimu-source-audit-02")
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    entry = next(f for f in metadata["files"] if f["key"] == "analysis.zip")
    if (entry["size"], entry["checksum"]) != (
        169608328, "md5:6c937c7122fee77ff1ce306175cb26c7"
    ):
        raise ValueError("source pin mismatch")
    path = output / "analysis.zip"
    if not args.verify_existing:
        import requests

        received = 0
        with requests.get(entry["links"]["self"], stream=True, timeout=30) as response:
            response.raise_for_status()
            with path.open("xb") as target:
                for chunk in response.iter_content(1024 * 1024):
                    received += len(chunk)
                    if received > entry["size"]:
                        raise ValueError("size exceeded")
                    target.write(chunk)
    verify_archive(path, size=entry["size"], checksum=entry["checksum"])
    write(output / "source.json", entry)
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
