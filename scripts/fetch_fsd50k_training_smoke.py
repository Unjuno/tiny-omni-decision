"""Fetch one rights-filtered, fixed-ID training clip per event from split ZIP."""

import csv
import hashlib
import io
import json
import struct
import urllib.request
import zipfile
import zlib
from pathlib import Path
from unittest.mock import patch

from scripts.fetch_starss23_smoke import HttpRangeReader
from scripts.run_starss23_ipv4 import IPv4HTTPSHandler

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
EVENTS = ("Bell", "Clapping", "Telephone", "Walk_and_footsteps",
          "Musical_instrument", "Knock", "Door", "Water_tap_and_faucet")
ALLOW = {"http://creativecommons.org/licenses/by/3.0/",
         "http://creativecommons.org/publicdomain/zero/1.0/"}


def central_entries(data, count):
    """Parse a directory-only envelope; preserve original disk and local offsets."""
    end = struct.pack("<4s4H2IH", b"PK\x05\x06", 0, 0, count, count, len(data), 0, 0)
    with zipfile.ZipFile(io.BytesIO(data + end)) as archive:
        entries = archive.infolist()
    if len(entries) != count or len({e.filename for e in entries}) != count:
        raise ValueError("invalid directory count or duplicate members")
    return {e.filename: e for e in entries}


def main():
    source = ROOT / "fsd50k-metadata-audit-01"
    output = ROOT / "fsd50k-training-smoke-02"
    output.mkdir(exist_ok=False)
    hashes = json.loads((source / "downloads.json").read_text())
    for name, info in hashes.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError("metadata hash mismatch")
    with zipfile.ZipFile(source / "FSD50K.metadata.zip") as z:
        meta = json.loads(z.read("FSD50K.metadata/dev_clips_info_FSD50K.json"))
    with zipfile.ZipFile(source / "FSD50K.ground_truth.zip") as z:
        rows = list(csv.DictReader(io.StringIO(z.read("FSD50K.ground_truth/dev.csv").decode())))
    selected, uploaders, ids = [], set(), set()
    for event in EVENTS:
        candidates = [r for r in rows if r["split"] == "train"
                      and event in r["labels"].split(",")
                      and meta[r["fname"]]["license"] in ALLOW
                      and meta[r["fname"]]["uploader"] not in uploaders and r["fname"] not in ids]
        candidates.sort(key=lambda r: hashlib.sha256(f"17:{r['fname']}".encode()).hexdigest())
        if not candidates:
            raise ValueError("missing eligible event")
        row = candidates[0]
        selected.append({"event": event, "id": row["fname"], "known_positive_labels": row["labels"],
                         "license": meta[row["fname"]]["license"],
                         "uploader": meta[row["fname"]]["uploader"]})
        ids.add(row["fname"])
        uploaders.add(meta[row["fname"]]["uploader"])
    (output / "selection.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    names = [f"FSD50K.dev_audio.z{i:02d}" for i in range(1, 6)] + ["FSD50K.dev_audio.zip"]
    readers = [HttpRangeReader(f"https://zenodo.org/api/records/4060432/files/{name}/content",
                              3221225472 if i < 5 else 2306663327, budget=32 * 1024**2)
               for i, name in enumerate(names)]

    def read_parts(disk, offset, count):
        chunks = []
        while count:
            reader = readers[disk]
            length = min(count, reader.size - offset)
            if length <= 0:
                raise ValueError("invalid split offset")
            reader.seek(offset)
            chunks.append(reader.read(length))
            count -= length
            disk, offset = disk + 1, 0
        return b"".join(chunks)

    result = {"status": "RUNNING", "selection": selected, "files": [],
              "whole_archive_md5_verified": False, "eval_opened": False}
    try:
        opener = urllib.request.build_opener(IPv4HTTPSHandler())
        with patch("urllib.request.urlopen", opener.open):
            readers[-1].seek(2306663327 - 22)
            end = struct.unpack("<4s4H2IH", readers[-1].read(22))
            if end != (b"PK\x05\x06", 5, 5, 40967, 40967, 3962717, 2302700512, 0):
                raise ValueError("directory footer differs from pinned probe")
            readers[-1].seek(end[6])
            directory = readers[-1].read(end[5])
            result["directory_sha256"] = hashlib.sha256(directory).hexdigest()
            index = central_entries(directory, end[4])
            for item in selected:
                name = f"FSD50K.dev_audio/{item['id']}.wav"
                info = index[name]
                if info.file_size > 32 * 1024**2 or info.compress_size > 16 * 1024**2:
                    raise ValueError("selected member exceeds smoke budget; no substitution")
                header = read_parts(info.volume, info.header_offset, 30)
                fields = struct.unpack("<4s5H3I2H", header)
                if fields[0] != b"PK\x03\x04" or fields[2] & 1 or fields[3] != info.compress_type:
                    raise ValueError("invalid or encrypted member")
                extra = fields[-2] + fields[-1]
                raw = read_parts(info.volume, info.header_offset, 30 + extra + info.compress_size)
                if raw[30:30 + fields[-2]].decode() != name:
                    raise ValueError("local/central name mismatch")
                data = raw[30 + extra:]
                if info.compress_type == 8:
                    data = zlib.decompress(data, -15)
                elif info.compress_type != 0:
                    raise ValueError("unsupported compression")
                if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
                    raise ValueError("ZIP integrity mismatch")
                (output / f"{item['id']}.wav").write_bytes(data)
                result["files"].append({"id": item["id"], "bytes": len(data),
                                        "sha256": hashlib.sha256(data).hexdigest(),
                                        "zip_crc32": info.CRC, "disk": info.volume})
                (output / "progress.json").write_text(json.dumps(result, indent=2))
        result["status"] = "PASS"
    except Exception as error:
        result["status"], result["error"] = "FAIL", repr(error)
        raise
    finally:
        result["transferred_bytes"] = sum(r.transferred for r in readers)
        (output / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
