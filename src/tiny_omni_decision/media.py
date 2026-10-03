from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

from .dataset import _speech_command_label, iter_hub_rows, sha256_file
from .schema import DatasetManifest, DecisionExample, MediaRef


class _HttpRangeReader(io.RawIOBase):
    """Seekable read-only HTTP range view for selecting a few ZIP members."""

    def __init__(self, url: str, chunk_size: int = 4 * 1024 * 1024) -> None:
        self.url = url
        self.chunk_size = chunk_size
        self.position = 0
        self.cache_start = -1
        self.cache = b""
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            self.size = int(response.headers["Content-Length"])
            self.range_supported = "bytes" in response.headers.get("Accept-Ranges", "").lower()
        if not self.range_supported:
            raise ValueError(f"remote source does not support byte ranges: {url}")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"invalid seek whence: {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self.position
        end = min(self.position + size, self.size)
        result = bytearray()
        while self.position < end:
            chunk_start = self.position // self.chunk_size * self.chunk_size
            if chunk_start != self.cache_start:
                chunk_end = min(chunk_start + self.chunk_size, self.size) - 1
                request = urllib.request.Request(
                    self.url, headers={"Range": f"bytes={chunk_start}-{chunk_end}"}
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    if response.status != 206:
                        raise OSError(f"range request returned HTTP {response.status}")
                    self.cache = response.read()
                self.cache_start = chunk_start
            offset = self.position - self.cache_start
            count = min(end - self.position, len(self.cache) - offset)
            if count <= 0:
                raise OSError("range response did not cover the requested ZIP data")
            result.extend(self.cache[offset : offset + count])
            self.position += count
        return bytes(result)


def materialize_clevr4_images(
    examples: list[DecisionExample], *, data_root: Path, archive_url: str
) -> tuple[list[DecisionExample], dict[str, object]]:
    requested: dict[str, Path] = {}
    for example in examples:
        for reference in example.media:
            if reference.kind != "image" or reference.path or not reference.uri:
                continue
            marker = "/images/"
            if (
                not reference.uri.startswith("source-ref://sgvaze/clevr4@")
                or marker not in reference.uri
            ):
                continue
            filename = reference.uri.rsplit(marker, 1)[1]
            if Path(filename).name != filename or not filename.endswith(".png"):
                raise ValueError(f"invalid Clevr-4 source image path: {reference.uri}")
            requested[filename] = data_root / "raw" / "clevr4-10k" / "images" / filename

    missing = {name: path for name, path in requested.items() if not path.is_file()}
    if missing:
        reader = _HttpRangeReader(archive_url)
        with zipfile.ZipFile(reader) as archive:
            for filename, path in missing.items():
                archive_name = f"images/{filename}"
                info = archive.getinfo(archive_name)
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, path.open("wb") as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
                with path.open("rb") as image_file:
                    if image_file.read(8) != b"\x89PNG\r\n\x1a\n":
                        path.unlink(missing_ok=True)
                        raise ValueError(f"Clevr-4 media member is not a PNG: {archive_name}")

    checksums = {name: sha256_file(path) for name, path in requested.items()}
    converted: list[DecisionExample] = []
    for example in examples:
        media: list[MediaRef] = []
        for reference in example.media:
            if reference.kind == "image" and reference.uri:
                filename = reference.uri.rsplit("/", 1)[1]
                if filename in requested:
                    media.append(
                        MediaRef(
                            kind="image",
                            path=requested[filename].relative_to(data_root).as_posix(),
                            sha256=checksums[filename],
                            license=reference.license,
                        )
                    )
                    continue
            media.append(reference)
        converted.append(example.model_copy(update={"media": media}))
    return converted, {
        "archive_url": archive_url,
        "archive_revision_or_checksum": (
            "SHA-512 pinned in the Clevr-4 dataset manifest; partial range reads "
            "do not verify the whole archive hash"
        ),
        "images_materialized": len(requested),
        "images_newly_downloaded": len(missing),
        "image_sha256": checksums,
    }


def materialize_clevrer_videos(
    examples: list[DecisionExample],
    *,
    data_root: Path,
    archive_urls: dict[str, str],
) -> tuple[list[DecisionExample], dict[str, object]]:
    requested: dict[tuple[str, str], Path] = {}
    for example in examples:
        for reference in example.media:
            if reference.kind != "video" or reference.path or not reference.uri:
                continue
            prefix = "source-ref://CLEVRER/"
            if not reference.uri.startswith(prefix):
                continue
            parts = reference.uri.removeprefix(prefix).split("/")
            if len(parts) != 4 or parts[1] != "videos" or parts[2] not in archive_urls:
                raise ValueError(f"invalid CLEVRER video reference: {reference.uri}")
            split, filename = parts[2], parts[3]
            if Path(filename).name != filename or not filename.endswith(".mp4"):
                raise ValueError(f"invalid CLEVRER video filename: {reference.uri}")
            requested[(split, filename)] = (
                data_root / "raw" / "clevrer" / "videos" / split / filename
            )

    missing_by_split: dict[str, dict[str, Path]] = {}
    for (split, filename), path in requested.items():
        if not path.is_file():
            missing_by_split.setdefault(split, {})[filename] = path
    for split, missing in missing_by_split.items():
        with zipfile.ZipFile(_HttpRangeReader(archive_urls[split])) as archive:
            names_by_basename = {
                name.rsplit("/", 1)[-1]: name
                for name in archive.namelist()
                if not name.endswith("/")
            }
            for filename, path in missing.items():
                archive_name = names_by_basename.get(filename)
                if archive_name is None:
                    raise FileNotFoundError(f"CLEVRER ZIP is missing {filename}")
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(archive_name) as source, path.open("wb") as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
                with path.open("rb") as video_file:
                    header = video_file.read(12)
                    if len(header) < 12 or header[4:8] != b"ftyp":
                        path.unlink(missing_ok=True)
                        raise ValueError(f"CLEVRER member is not an MP4: {archive_name}")

    hashes = {f"{split}/{name}": sha256_file(path) for (split, name), path in requested.items()}
    converted = []
    for example in examples:
        media = []
        for reference in example.media:
            if (
                reference.kind == "video"
                and reference.uri
                and reference.uri.startswith("source-ref://CLEVRER/")
            ):
                parts = reference.uri.split("/")
                split, filename = parts[-2], parts[-1]
                path = requested[(split, filename)]
                media.append(
                    MediaRef(
                        kind="video",
                        path=path.relative_to(data_root).as_posix(),
                        sha256=hashes[f"{split}/{filename}"],
                        license=reference.license,
                    )
                )
            else:
                media.append(reference)
        converted.append(example.model_copy(update={"media": media}))
    return converted, {
        "archive_urls": archive_urls,
        "archive_sizes_bytes": {
            split: _HttpRangeReader(url).size for split, url in archive_urls.items()
        },
        "videos_materialized": len(requested),
        "videos_newly_downloaded": sum(len(items) for items in missing_by_split.values()),
        "video_sha256": hashes,
        "archive_hash_note": (
            "Selected members extracted by HTTP byte-range reads; "
            "full archive hashes were not computed."
        ),
    }


def materialize_speech_commands_audio(
    examples: list[DecisionExample],
    *,
    manifest: DatasetManifest,
    data_root: Path,
) -> tuple[list[DecisionExample], dict[str, object]]:
    requested: dict[tuple[str, str], Path] = {}
    for example in examples:
        if example.modality != "audio":
            continue
        parts = example.source_record_id.replace("\\", "/").split("/")
        if len(parts) != 2 or parts[0] != example.target or not parts[1].endswith(".wav"):
            raise ValueError(f"invalid Speech Commands record id: {example.source_record_id}")
        label, filename = parts
        requested[(label, filename)] = (
            data_root / "raw" / "speech-commands" / manifest.split / label / filename
        )

    missing = {key for key, path in requested.items() if not path.is_file()}
    found: set[tuple[str, str]] = set()
    if missing:
        for row in iter_hub_rows(manifest):
            audio = row.get("audio")
            if not isinstance(audio, dict):
                continue
            label = _speech_command_label(row)
            path_value = audio.get("path")
            payload = audio.get("bytes")
            if not label or not path_value or not isinstance(payload, bytes):
                continue
            key = (label, Path(str(path_value)).name)
            if key not in missing:
                continue
            path = requested[key]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            with path.open("rb") as audio_file:
                header = audio_file.read(12)
                if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
                    path.unlink(missing_ok=True)
                    raise ValueError(f"Speech Commands media is not a WAV file: {key}")
            found.add(key)
            if found == missing:
                break
    if found != missing:
        raise FileNotFoundError(
            f"Speech Commands audio bytes missing for {sorted(missing - found)}"
        )

    hashes = {f"{label}/{name}": sha256_file(path) for (label, name), path in requested.items()}
    converted = []
    for example in examples:
        if example.modality != "audio":
            converted.append(example)
            continue
        label, filename = example.source_record_id.replace("\\", "/").split("/")
        path = requested[(label, filename)]
        reference = example.media[0]
        converted.append(
            example.model_copy(
                update={
                    "media": [
                        MediaRef(
                            kind="audio",
                            path=path.relative_to(data_root).as_posix(),
                            sha256=hashes[f"{label}/{filename}"],
                            license=reference.license,
                        )
                    ]
                }
            )
        )
    return converted, {
        "source_dataset": manifest.dataset_id,
        "source_revision": manifest.revision,
        "parquet_files": manifest.notes["parquet_files"],
        "audio_materialized": len(requested),
        "audio_newly_downloaded": len(found),
        "audio_sha256": hashes,
    }
