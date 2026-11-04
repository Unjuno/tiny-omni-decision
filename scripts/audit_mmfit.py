"""Read source metadata and ZIP documentation only; no sensor/media payloads."""

import io
import json
import zipfile
from pathlib import Path

import requests

from scripts.range_generalization import sha, write


class RemoteZip(io.RawIOBase):
    def __init__(self, url, size, etag):
        self.url, self.size, self.etag, self.position = url, size, etag, 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        size = min(size if size >= 0 else self.size, self.size - self.position)
        if size == 0:
            return b""
        if size < 0 or size > 4_000_000:
            raise ValueError("metadata range exceeds budget")
        end = self.position + size - 1
        r = requests.get(self.url, headers={"Range": f"bytes={self.position}-{end}",
                                           "If-Match": self.etag}, timeout=30)
        r.raise_for_status()
        if r.status_code != 206 or len(r.content) != size:
            raise ValueError("range response mismatch")
        if r.headers.get("Content-Range") != f"bytes {self.position}-{end}/{self.size}":
            raise ValueError("content range mismatch")
        self.position += size
        return r.content


def main():
    output = Path("C:/CodexArtifacts/pretrained-reuse-r2/mmfit-source-audit-01")
    output.mkdir(exist_ok=False)
    r = requests.get("https://zenodo.org/api/records/7672767", timeout=30)
    r.raise_for_status()
    write(output / "video-record.json", r.json())
    url = "https://s3.eu-west-2.amazonaws.com/vradu.uk/mm-fit.zip"
    head = requests.head(url, timeout=30)
    head.raise_for_status()
    write(output / "sensor-headers.json", dict(head.headers))
    remote = RemoteZip(url, int(head.headers["Content-Length"]), head.headers["ETag"])
    docs = []
    with zipfile.ZipFile(remote) as z:
        write(output / "zip-inventory.json", [
            {"name": i.filename, "bytes": i.file_size, "crc": i.CRC} for i in z.infolist()
        ])
        for i in z.infolist():
            if i.filename.lower().endswith((".txt", ".md")) and i.file_size < 256_000:
                path = output / f"document-{len(docs)}.txt"
                path.write_bytes(z.read(i))
                docs.append({"member": i.filename, "saved": str(path), "sha256": sha(path)})
    write(output / "result.json", {"video_license": r.json()["metadata"]["license"],
                                   "sensor_documentation": docs,
                                   "sensor_arrays_or_media_read": False})
    print(json.dumps(docs))


if __name__ == "__main__":
    main()
