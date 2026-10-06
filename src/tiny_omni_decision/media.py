from __future__ import annotations

import io
import os
import struct
import tempfile
import urllib.request
import zipfile
import zlib
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from urllib.parse import unquote

from .dataset import _speech_command_label, iter_hub_rows, sha256_file
from .physionpp import (
    PHYSIONPP_ETAG,
    PHYSIONPP_LAST_MODIFIED,
    PHYSIONPP_READOUT_URL,
    PHYSIONPP_REVISION,
)
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


def _is_valid_png(path: Path) -> bool:
    """Reject partial or malformed PNGs left by interrupted media extraction."""
    try:
        with path.open("rb") as image_file:
            if image_file.read(8) != b"\x89PNG\r\n\x1a\n":
                return False
            saw_header = False
            saw_image_data = False
            image_data_closed = False
            inflater: zlib.Decompress | None = None
            while True:
                length_bytes = image_file.read(4)
                if len(length_bytes) != 4:
                    return False
                length = struct.unpack(">I", length_bytes)[0]
                if length > 64 * 1024 * 1024:
                    return False
                chunk_type = image_file.read(4)
                chunk_data = image_file.read(length)
                crc_bytes = image_file.read(4)
                if len(chunk_type) != 4 or len(chunk_data) != length or len(crc_bytes) != 4:
                    return False
                expected_crc = struct.unpack(">I", crc_bytes)[0]
                if zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF != expected_crc:
                    return False

                if not saw_header:
                    if chunk_type != b"IHDR" or length != 13:
                        return False
                    width, height = struct.unpack(">II", chunk_data[:8])
                    if width == 0 or height == 0:
                        return False
                    saw_header = True
                    continue
                if chunk_type == b"IHDR":
                    return False
                if chunk_type == b"IDAT":
                    if image_data_closed:
                        return False
                    if inflater is None:
                        inflater = zlib.decompressobj()
                    inflater.decompress(chunk_data)
                    saw_image_data = True
                    continue
                if saw_image_data:
                    image_data_closed = True
                if chunk_type == b"IEND":
                    return (
                        length == 0
                        and saw_image_data
                        and inflater is not None
                        and inflater.eof
                        and not inflater.unused_data
                        and image_file.read(1) == b""
                    )
    except (OSError, ValueError, OverflowError, zlib.error):
        return False


def materialize_clevr4_images(
    examples: list[DecisionExample],
    *,
    data_root: Path,
    archive_url: str,
    archive_path: Path | None = None,
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

    missing = {name: path for name, path in requested.items() if not _is_valid_png(path)}
    if missing:
        archive_source: Path | _HttpRangeReader
        if archive_path is not None:
            archive_source = archive_path
        else:
            archive_source = _HttpRangeReader(archive_url)
        with zipfile.ZipFile(archive_source) as archive:
            for filename, path in sorted(
                missing.items(),
                key=lambda item: archive.getinfo(f"images/{item[0]}").header_offset,
            ):
                archive_name = f"images/{filename}"
                info = archive.getinfo(archive_name)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(path.suffix + ".part")
                with archive.open(info) as source, temporary.open("wb") as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
                if not _is_valid_png(temporary):
                    raise ValueError(f"Clevr-4 media member is not a valid PNG: {archive_name}")
                temporary.replace(path)

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
            "SHA-512 pinned in the Clevr-4 dataset manifest; full archive hash "
            "verified before local materialization"
            if archive_path is not None
            else "SHA-512 pinned in the Clevr-4 dataset manifest; partial range reads "
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
            members_by_basename = {
                info.filename.rsplit("/", 1)[-1]: info
                for info in archive.infolist()
                if not info.is_dir()
            }
            absent = set(missing) - set(members_by_basename)
            if absent:
                raise FileNotFoundError(f"CLEVRER ZIP is missing {sorted(absent)}")
            for filename, path in sorted(
                missing.items(),
                key=lambda item: members_by_basename[item[0]].header_offset,
            ):
                info = members_by_basename[filename]
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, path.open("wb") as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
                with path.open("rb") as video_file:
                    header = video_file.read(12)
                    if len(header) < 12 or header[4:8] != b"ftyp":
                        path.unlink(missing_ok=True)
                        raise ValueError(f"CLEVRER member is not an MP4: {info.filename}")

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


def materialize_physionpp_videos(
    examples: list[DecisionExample],
    *,
    data_root: Path,
    archive_url: str = PHYSIONPP_READOUT_URL,
) -> tuple[list[DecisionExample], dict[str, object]]:
    """Extract pinned RGB videos and verify frame counts or explicit cutoffs."""
    prefix = f"source-ref://physionpp-readout/{PHYSIONPP_REVISION}/"
    requested: dict[str, tuple[Path, Path, int | None, int | None]] = {}
    for example in examples:
        for reference in example.media:
            if reference.kind != "video" or reference.path or not reference.uri:
                continue
            if reference.uri.startswith("source-ref://physionpp-readout/"):
                if not reference.uri.startswith(prefix):
                    raise ValueError(f"Physion++ source revision is not pinned: {reference.uri}")
                member = unquote(reference.uri.removeprefix(prefix))
                parts = member.split("/")
                if (
                    len(parts) < 4
                    or parts[0] != "readout_data_v1"
                    or any(part in {"", ".", ".."} for part in parts)
                    or not parts[-1].endswith("_img.mp4")
                ):
                    raise ValueError(f"invalid Physion++ RGB member: {reference.uri}")
                if reference.num_frames is None:
                    raise ValueError(
                        f"Physion++ video lacks its pinned frame count: {reference.uri}"
                    )
                raw_path = data_root / "raw" / "physionpp-readout" / Path(*parts)
                final_path = (
                    data_root / "processed" / "physionpp-readout" / Path(*parts)
                    if reference.end_frame is not None
                    else raw_path
                )
                requested[member] = (
                    raw_path,
                    final_path,
                    reference.end_frame,
                    reference.num_frames,
                )

    if not requested:
        return examples, {
            "archive_url": archive_url,
            "videos_materialized": 0,
            "video_end_frame": {},
            "video_num_frames": {},
            "video_sha256": {},
        }

    request = urllib.request.Request(archive_url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        size = int(response.headers["Content-Length"])
        etag = response.headers.get("ETag")
        last_modified = response.headers.get("Last-Modified")
    if size != 2_971_891_321 or etag != PHYSIONPP_ETAG or last_modified != PHYSIONPP_LAST_MODIFIED:
        raise ValueError("Physion++ archive does not match the pinned source object")

    downloaded = 0
    existing_verified = 0
    with zipfile.ZipFile(_HttpRangeReader(archive_url, chunk_size=512 * 1024)) as archive:
        members_by_name = {info.filename: info for info in archive.infolist() if not info.is_dir()}
        absent = set(requested) - members_by_name.keys()
        if absent:
            raise FileNotFoundError(f"Physion++ ZIP is missing {sorted(absent)}")
        for member, path in sorted(
            requested.items(), key=lambda item: members_by_name[item[0]].header_offset
        ):
            raw_path, final_path, end_frame, source_num_frames = path
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            final_path.parent.mkdir(parents=True, exist_ok=True)
            if not raw_path.is_file():
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{raw_path.name}.", suffix=".tmp", dir=raw_path.parent
                )
                os.close(descriptor)
                temporary_path = Path(temporary_name)
                try:
                    with (
                        archive.open(members_by_name[member]) as source,
                        temporary_path.open("wb") as target,
                    ):
                        while chunk := source.read(1024 * 1024):
                            target.write(chunk)
                    with temporary_path.open("rb") as video_file:
                        header = video_file.read(12)
                    if len(header) < 12 or header[4:8] != b"ftyp":
                        raise ValueError(f"Physion++ member is not an MP4: {member}")
                    temporary_path.replace(raw_path)
                    downloaded += 1
                except Exception:
                    temporary_path.unlink(missing_ok=True)
                    raise
            else:
                info = members_by_name[member]
                if raw_path.stat().st_size != info.file_size:
                    raise ValueError(f"existing Physion++ member has the wrong size: {raw_path}")
                with raw_path.open("rb") as video_file:
                    header = video_file.read(12)
                if len(header) < 12 or header[4:8] != b"ftyp":
                    raise ValueError(f"existing Physion++ media is not an MP4: {raw_path}")
                crc = 0
                with raw_path.open("rb") as video_file:
                    while chunk := video_file.read(1024 * 1024):
                        crc = zlib.crc32(chunk, crc)
                if crc & 0xFFFFFFFF != info.CRC:
                    raise ValueError(
                        f"existing Physion++ member failed ZIP CRC validation: {raw_path}"
                    )
                existing_verified += 1

            if end_frame is not None and not final_path.is_file():
                _truncate_physionpp_video(raw_path, final_path, end_frame=end_frame)
            actual_frames = _video_frame_count(final_path)
            expected_frames = end_frame if end_frame is not None else source_num_frames
            if actual_frames != expected_frames:
                raise ValueError(
                    f"Physion++ video has {actual_frames} frames; expected {expected_frames}: "
                    f"{final_path}"
                )

    raw_hashes = {member: sha256_file(paths[0]) for member, paths in requested.items()}
    hashes = {member: sha256_file(paths[1]) for member, paths in requested.items()}
    converted = []
    for example in examples:
        media = []
        for reference in example.media:
            if reference.kind == "video" and reference.uri and reference.uri.startswith(prefix):
                member = unquote(reference.uri.removeprefix(prefix))
                _, path, end_frame, source_num_frames = requested[member]
                media.append(
                    MediaRef(
                        kind="video",
                        path=path.relative_to(data_root).as_posix(),
                        sha256=hashes[member],
                        license=reference.license,
                        num_frames=end_frame if end_frame is not None else source_num_frames,
                    )
                )
            else:
                media.append(reference)
        converted.append(example.model_copy(update={"media": media}))
    return converted, {
        "archive_url": archive_url,
        "archive_size_bytes": size,
        "archive_etag": etag,
        "archive_last_modified": last_modified,
        "videos_materialized": len(requested),
        "videos_newly_downloaded": downloaded,
        "videos_existing_verified": existing_verified,
        "video_end_frame": {member: paths[2] for member, paths in requested.items()},
        "video_num_frames": {
            member: paths[2] if paths[2] is not None else paths[3]
            for member, paths in requested.items()
        },
        "raw_video_sha256": raw_hashes,
        "video_sha256": hashes,
        "archive_hash_note": (
            "Only selected RGB members were extracted with HTTP byte ranges; "
            "the full archive was not downloaded or hashed."
        ),
    }


def _video_frame_count(path: Path) -> int:
    import av

    with av.open(str(path), mode="r") as container:
        streams = container.streams.video
        if len(streams) != 1:
            raise ValueError(f"Physion++ video must have one video stream: {path}")
        return sum(1 for _ in container.decode(streams[0]))


def _truncate_physionpp_video(source: Path, destination: Path, *, end_frame: int) -> None:
    """Re-encode exactly the visible prefix, excluding the outcome after the cutoff."""
    import av

    if end_frame < 1:
        raise ValueError("Physion++ video cutoff must be a positive frame index")
    temporary_path = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.mp4")
    try:
        with av.open(str(source), mode="r") as input_container:
            streams = input_container.streams.video
            if len(streams) != 1:
                raise ValueError(f"Physion++ source must have one video stream: {source}")
            input_stream = streams[0]
            rate = input_stream.average_rate or input_stream.base_rate
            if rate is None or rate <= 0:
                raise ValueError(f"Physion++ source has no valid frame rate: {source}")
            with av.open(str(temporary_path), mode="w", format="mp4") as output_container:
                output_stream = output_container.add_stream("libx264", rate=rate)
                output_stream.width = input_stream.codec_context.width
                output_stream.height = input_stream.codec_context.height
                output_stream.pix_fmt = "yuv420p"
                output_stream.time_base = Fraction(rate.denominator, rate.numerator)
                seen = 0
                for frame in input_container.decode(input_stream):
                    if seen >= end_frame:
                        break
                    frame = frame.reformat(
                        width=output_stream.width,
                        height=output_stream.height,
                        format=output_stream.pix_fmt,
                    )
                    frame.pts = seen
                    frame.time_base = output_stream.time_base
                    for packet in output_stream.encode(frame):
                        output_container.mux(packet)
                    seen += 1
                if seen != end_frame:
                    raise ValueError(
                        f"Physion++ source has only {seen} decoded frames before requested "
                        f"cutoff {end_frame}: {source}"
                    )
                for packet in output_stream.encode(None):
                    output_container.mux(packet)
        with temporary_path.open("rb") as video_file:
            header = video_file.read(12)
        if len(header) < 12 or header[4:8] != b"ftyp":
            raise ValueError(f"Physion++ cropped output is not a valid MP4: {temporary_path}")
        temporary_path.replace(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def materialize_speech_commands_audio(
    examples: list[DecisionExample],
    *,
    manifest: DatasetManifest | Mapping[str, DatasetManifest],
    data_root: Path,
) -> tuple[list[DecisionExample], dict[str, object]]:
    manifests_by_split = (
        {manifest.split: manifest} if isinstance(manifest, DatasetManifest) else dict(manifest)
    )
    for split, source_manifest in manifests_by_split.items():
        if source_manifest.split != split:
            raise ValueError(
                f"Speech Commands manifest key {split!r} does not match "
                f"its declared split {source_manifest.split!r}"
            )

    requested: dict[tuple[str, str, str], Path] = {}
    for example in examples:
        if example.modality != "audio":
            continue
        source_manifest = manifests_by_split.get(example.split)
        if source_manifest is None:
            raise ValueError(f"no Speech Commands manifest supplied for split {example.split!r}")
        if example.source != source_manifest.dataset_id:
            raise ValueError(
                f"Speech Commands source mismatch for {example.id}: "
                f"{example.source!r} != {source_manifest.dataset_id!r}"
            )
        if example.source_revision != source_manifest.revision:
            raise ValueError(
                f"Speech Commands revision mismatch for {example.id}: "
                f"{example.source_revision!r} != {source_manifest.revision!r}"
            )
        if example.split != source_manifest.split:
            raise ValueError(
                f"Speech Commands split mismatch for {example.id}: "
                f"{example.split!r} != {source_manifest.split!r}"
            )
        parts = example.source_record_id.replace("\\", "/").split("/")
        if len(parts) != 2 or parts[0] != example.target or not parts[1].endswith(".wav"):
            raise ValueError(f"invalid Speech Commands record id: {example.source_record_id}")
        label, filename = parts
        requested[(example.split, label, filename)] = (
            data_root / "raw" / "speech-commands" / example.split / label / filename
        )

    missing_by_split: dict[str, set[tuple[str, str, str]]] = {}
    for key, path in requested.items():
        if not path.is_file():
            missing_by_split.setdefault(key[0], set()).add(key)
    found: set[tuple[str, str, str]] = set()
    for split, missing in sorted(missing_by_split.items()):
        split_found: set[tuple[str, str, str]] = set()
        source_manifest = manifests_by_split[split]
        for row in iter_hub_rows(source_manifest):
            audio = row.get("audio")
            if not isinstance(audio, dict):
                continue
            label = _speech_command_label(row)
            path_value = audio.get("path")
            payload = audio.get("bytes")
            if not label or not path_value or not isinstance(payload, bytes):
                continue
            key = (split, label, Path(str(path_value)).name)
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
            split_found.add(key)
            if split_found == missing:
                break
        if split_found != missing:
            absent = sorted(missing - split_found)
            raise FileNotFoundError(f"Speech Commands {split} audio bytes missing for {absent}")

    for key, path in requested.items():
        with path.open("rb") as audio_file:
            header = audio_file.read(12)
        if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise ValueError(f"Speech Commands media is not a WAV file: {key}")

    hashes = {
        f"{split}/{label}/{name}": sha256_file(path)
        for (split, label, name), path in requested.items()
    }
    converted = []
    for example in examples:
        if example.modality != "audio":
            converted.append(example)
            continue
        label, filename = example.source_record_id.replace("\\", "/").split("/")
        key = (example.split, label, filename)
        path = requested[key]
        reference = example.media[0]
        converted.append(
            example.model_copy(
                update={
                    "media": [
                        MediaRef(
                            kind="audio",
                            path=path.relative_to(data_root).as_posix(),
                            sha256=hashes[f"{example.split}/{label}/{filename}"],
                            license=reference.license,
                        )
                    ]
                }
            )
        )
    manifests_metadata = {
        split: {
            "source_dataset": source_manifest.dataset_id,
            "source_revision": source_manifest.revision,
            "parquet_files": source_manifest.notes.get("parquet_files", []),
        }
        for split, source_manifest in sorted(manifests_by_split.items())
    }
    return converted, {
        "source_manifests_by_split": manifests_metadata,
        "audio_materialized": len(requested),
        "audio_newly_downloaded": len(found),
        "audio_sha256": hashes,
    }
