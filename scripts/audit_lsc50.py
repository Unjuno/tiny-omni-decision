"""Pin LSC50 license and inspect paired archive names without reading recordings."""

import json
import zipfile
from pathlib import Path

import requests

from scripts.audit_mmfit import RemoteZip
from scripts.range_generalization import write


def main():
    output = Path("C:/CodexArtifacts/pretrained-reuse-r2/lsc50-source-audit-02")
    output.mkdir(exist_ok=False)
    r = requests.get("https://api.figshare.com/v2/articles/27383016/versions/1", timeout=30)
    r.raise_for_status()
    metadata = r.json()
    if metadata["license"]["name"] != "CC BY 4.0":
        raise ValueError("unexpected license")
    write(output / "metadata.json", metadata)
    summaries = []
    for f in metadata["files"]:
        if f["name"] not in ("IMU.zip", "VIDEOS.zip"):
            continue
        head = requests.get(f["download_url"], headers={"Range": "bytes=0-0"}, timeout=30)
        head.raise_for_status()
        if head.status_code != 206 or head.headers.get("Content-Range") != f'bytes 0-0/{f["size"]}':
            raise ValueError("archive size differs from version metadata")
        write(output / (f["name"] + ".headers.json"), dict(head.headers))
        remote = RemoteZip(f["download_url"], f["size"], head.headers["ETag"])
        with zipfile.ZipFile(remote) as z:
            inventory = [{"name": i.filename, "size": i.file_size, "crc": i.CRC}
                         for i in z.infolist()]
        write(output / (f["name"] + ".inventory.json"), inventory)
        summaries.append({"archive": f["name"], "members": len(inventory),
                          "examples": inventory[:12]})
    write(output / "result.json", {"archives": summaries, "recordings_read": False})
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
