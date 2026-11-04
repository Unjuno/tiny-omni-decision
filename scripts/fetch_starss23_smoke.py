"""Fetch two room-distinct STARSS23 dev-train pairs using bounded HTTP ranges.

Only ZIP directory metadata is read for dev-test; its members are never fetched.
Whole media archive MD5 is not verified by a partial download. Each selected
member is checked by ZIP CRC and recorded with a local SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import urllib.request
import wave
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

BASE = "https://zenodo.org/api/records/7880637/files"
PINNED = {
    "LICENSE": "af4138d4fe3ccd91970f7ac1f22ae3f513ac570468a26f0f516540b5443b8745",
    "README.md": "9401015b18d0bbb2214668f0fc5a49f82005124d036808b284ccd27678c21716",
    "metadata_dev.zip": "2fa848545a520b4a9d490459de7f079aa49ae16daf0affc859c016f5146981d6",
}
ARCHIVES = {
    "foa_dev.zip": (3406085425, "316b834ee6393c22862c314cc3c7ebb0"),
    "video_dev.zip": (4102871214, "b92295e162600adb545ed99bdc0a9c76"),
}
NAME = re.compile(r"metadata_dev/dev-(train|test)-(sony|tau)/(fold\d+_(room\d+)_mix\d+)\.csv")


class HttpRangeReader(io.RawIOBase):
    def __init__(self, url: str, size: int, budget: int = 256 * 1024 * 1024, retries: int = 0):
        self.url, self.size, self.budget = url, size, budget
        self.retries = retries
        self.position = 0
        self.transferred = 0

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if whence not in (0, 1, 2) or position < 0 or position > self.size:
            raise ValueError("invalid range seek")
        self.position = position
        return position

    def read(self, size=-1):
        count = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if count == 0:
            return b""
        if count < 0 or self.transferred + count > self.budget:
            raise ValueError("range transfer budget exceeded")
        end = self.position + count - 1
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.position}-{end}"}
        )
        for attempt in range(self.retries + 1):
            if self.transferred + count > self.budget:
                raise ValueError("range transfer budget exceeded")
            with urllib.request.urlopen(request, timeout=60) as response:
                expected = f"bytes {self.position}-{end}/{self.size}"
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise ValueError("server did not return the exact requested byte range")
                data = response.read(count + 1)
            self.transferred += len(data)
            if len(data) == count:
                break
            if len(data) > count or attempt == self.retries:
                raise ValueError("range length mismatch")
        self.position += count
        return data


def metadata_inventory(names):
    rows = []
    for name in names:
        match = NAME.fullmatch(name)
        if match:
            split, site, clip, room = match.groups()
            rows.append(
                {
                    "member": name,
                    "split": split,
                    "site": site,
                    "clip": clip,
                    "room": f"{site}:{room}",
                }
            )
    if len({r["clip"] for r in rows}) != len(rows):
        raise ValueError("duplicate recording identity")
    train = {r["room"] for r in rows if r["split"] == "train"}
    test = {r["room"] for r in rows if r["split"] == "test"}
    if train & test:
        raise ValueError("room overlaps train and dev-test")
    return sorted(rows, key=lambda r: r["clip"])


def archive_index(archive, suffix):
    entries = {}
    for item in archive.infolist():
        if item.filename.endswith(suffix) and not item.filename.startswith("__MACOSX/"):
            stem = PurePosixPath(item.filename).stem
            if stem in entries:
                raise ValueError("duplicate media basename")
            entries[stem] = item
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name, expected in PINNED.items():
        if hashlib.sha256((args.source_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"source hash mismatch: {name}")
    report = {
        "record": "doi:10.5281/zenodo.7880637",
        "license": "MIT",
        "source_sha256": PINNED,
        "status": "RUNNING",
        "pairs": [],
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dev_test_contents_read": False,
        "official_eval_accessed": False,
    }
    readers = {
        name: HttpRangeReader(f"{BASE}/{name}/content", size)
        for name, (size, _) in ARCHIVES.items()
    }
    try:
        with (
            zipfile.ZipFile(args.source_dir / "metadata_dev.zip") as metadata,
            zipfile.ZipFile(readers["foa_dev.zip"]) as audio,
            zipfile.ZipFile(readers["video_dev.zip"]) as video,
        ):
            records = metadata_inventory(metadata.namelist())
            ai, vi = archive_index(audio, ".wav"), archive_index(video, ".mp4")
            report["record_counts"] = dict(Counter(r["split"] for r in records))
            report["room_groups"] = {
                s: sorted({r["room"] for r in records if r["split"] == s})
                for s in ("train", "test")
            }
            report["missing_video_clips"] = [r["clip"] for r in records if r["clip"] not in vi]
            selected = []
            for row in records:
                if row["split"] != "train" or row["clip"] not in ai or row["clip"] not in vi:
                    continue
                if row["room"] in {r["room"] for r in selected}:
                    continue
                selected.append(row)
                if len(selected) == 2:
                    break
            if len(selected) != 2:
                raise ValueError("two paired train rooms unavailable")
            import av

            for row in selected:
                pair = {**row, "files": {}}
                for kind, archive, entry in [
                    ("audio", audio, ai[row["clip"]]),
                    ("video", video, vi[row["clip"]]),
                ]:
                    if max(entry.file_size, entry.compress_size) > 128 * 1024 * 1024:
                        raise ValueError("selected member exceeds bounded smoke size")
                    data = archive.read(entry)  # ZipFile verifies CRC before returning.
                    path = output / PurePosixPath(entry.filename).name
                    path.write_bytes(data)
                    pair["files"][kind] = {
                        "path": str(path),
                        "member": entry.filename,
                        "bytes": len(data),
                        "crc32": entry.CRC,
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                labels = metadata.read(row["member"])
                (output / f"{row['clip']}.csv").write_bytes(labels)
                pair["annotation_sha256"] = hashlib.sha256(labels).hexdigest()
                with wave.open(pair["files"]["audio"]["path"]) as wav:
                    pair["audio_format"] = {
                        "channels": wav.getnchannels(),
                        "sample_rate": wav.getframerate(),
                        "seconds": wav.getnframes() / wav.getframerate(),
                    }
                with av.open(pair["files"]["video"]["path"]) as clip:
                    stream = clip.streams.video[0]
                    first = next(clip.decode(stream))
                    pair["video_format"] = {
                        "shape": list(first.to_ndarray(format="rgb24").shape),
                        "fps": float(stream.average_rate),
                        "seconds": float(stream.duration * stream.time_base),
                    }
                report["pairs"].append(pair)
            report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["archive_transfers"] = {
            name: {
                "bytes": reader.transferred,
                "upstream_md5": ARCHIVES[name][1],
                "whole_archive_md5_verified": False,
            }
            for name, reader in readers.items()
        }
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
