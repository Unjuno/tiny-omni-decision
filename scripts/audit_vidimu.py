"""Inspect VIDIMU archive structure and synchronization documentation only."""

import hashlib
import json
import zipfile
from pathlib import Path

import requests

from scripts.range_generalization import sha, write


def main():
    output = Path("C:/CodexArtifacts/pretrained-reuse-r2/vidimu-source-audit-02")
    output.mkdir(exist_ok=False)
    r = requests.get("https://zenodo.org/api/records/8210563", timeout=30)
    r.raise_for_status()
    metadata = r.json()
    if metadata["metadata"]["license"]["id"] != "cc-by-4.0":
        raise ValueError("license mismatch")
    write(output / "metadata.json", metadata)
    summaries = []
    for name in ("dataset.zip",):
        f = next(f for f in metadata["files"] if f["key"] == name)
        url = f["links"]["self"]
        path = output / name
        digest = hashlib.md5()
        received = 0
        with requests.get(url, stream=True, timeout=30) as download:
            download.raise_for_status()
            with path.open("xb") as target:
                for chunk in download.iter_content(1024 * 1024):
                    received += len(chunk)
                    if received > f["size"]:
                        raise ValueError("download exceeds pinned size")
                    digest.update(chunk)
                    target.write(chunk)
        if received != f["size"] or "md5:" + digest.hexdigest() != f["checksum"]:
            raise ValueError("whole archive checksum mismatch")
        with zipfile.ZipFile(path) as archive:
            inventory = [{"name": i.filename, "size": i.file_size, "crc": i.CRC}
                         for i in archive.infolist()]
            write(output / (name + ".inventory.json"), inventory)
            docs = []
            for i in archive.infolist():
                if "readme" in i.filename.lower() and i.file_size < 100_000:
                    path = output / f"{name}-readme-{len(docs)}.txt"
                    path.write_bytes(archive.read(i))
                    docs.append({"member": i.filename, "sha256": sha(path)})
        summaries.append({"archive": name, "members": len(inventory), "readmes": docs,
                          "sync_paths": [i["name"] for i in inventory
                                         if "infotosync" in i["name"].lower()]})
        print(json.dumps(summaries[-1]), flush=True)
    write(output / "result.json", {"record": 8210563, "license": "CC-BY-4.0",
                                   "archives": summaries, "raw_sensor_contents_decoded": False,
                                   "archive_sha256": sha(output / "dataset.zip")})


if __name__ == "__main__":
    main()
