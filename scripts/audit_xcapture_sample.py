"""Fetch and integrity-check a tiny train/validation-only X-Capture sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests

SOURCE_REPORT = Path("manifests/pretrained_reuse/xcapture-source-screen-20261128.json")
SPLIT_MANIFEST = Path("manifests/pretrained_reuse/xcapture-object-split-seed17.json")
DEFAULT_OUTPUT = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2-local/"
    "xcapture-sample-audit-20261128-attempt-01"
)
REPO_ID = "swistreich/XCapture"
REVISION = "b4468bf999aac250f34ef1b02459d902492a068f"
ARCHIVE_PATH = "XCapture_data.zip"
ARCHIVE_URL = (
    f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}/{ARCHIVE_PATH}"
)
EXPECTED_ARCHIVE_SHA256 = (
    "71855c2d9cdfb6d54ed379ff4059b6c370e6f7ef040830b0ac19063159c477c2"
)
EXPECTED_CENTRAL_SHA256 = (
    "3d8e546f1ca55e52a2b8e6c72aab99d14815e9da631589cb4d144f1179790f17"
)
MODALITY_BASENAMES = {
    "image": "rgb.png",
    "audio": "audio_normalized.wav",
    "tactile": "10N_tactile.png",
}
MODALITY_PATHS = {
    "image": "vision/rgb.png",
    "audio": "audio/audio_normalized.wav",
    "tactile": "tactile/10N_tactile.png",
}
AUXILIARY_PATHS = {"object_mask": "point_cloud/mask.png"}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _http_range(session: requests.Session, start: int, end: int) -> bytes:
    response = session.get(
        ARCHIVE_URL,
        headers={"Range": f"bytes={start}-{end}"},
        timeout=90,
    )
    response.raise_for_status()
    expected_range = f"bytes {start}-{end}/"
    if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(
        expected_range
    ):
        raise RuntimeError("pinned archive server did not honor the byte range")
    data = response.content
    if len(data) != end - start + 1:
        raise RuntimeError("archive range response length did not match request")
    return data


def _central_directory(session: requests.Session) -> tuple[bytes, int, int]:
    tail = session.get(
        ARCHIVE_URL, headers={"Range": "bytes=-65536"}, timeout=90
    )
    tail.raise_for_status()
    if tail.status_code != 206:
        raise RuntimeError("server did not provide a ZIP suffix range")
    offset = tail.content.rfind(b"PK\x06\x06")
    if offset < 0:
        raise RuntimeError("ZIP64 end-of-central-directory record not found")
    fields = struct.unpack_from("<4sQ2H2I4Q", tail.content, offset)
    entry_count, directory_size, directory_offset = fields[-4], fields[-2], fields[-1]
    end = directory_offset + directory_size - 1
    directory = _http_range(session, directory_offset, end)
    if _sha256(directory) != EXPECTED_CENTRAL_SHA256:
        raise RuntimeError("central-directory hash differs from the pinned source audit")
    return directory, entry_count, directory_offset


def _parse_entries(directory: bytes) -> dict[str, dict[str, int]]:
    entries: dict[str, dict[str, int]] = {}
    offset = 0
    while offset < len(directory):
        fields = struct.unpack_from("<4s6H3I5H2I", directory, offset)
        if fields[0] != b"PK\x01\x02":
            raise RuntimeError(f"invalid ZIP central-directory record at byte {offset}")
        method, crc, compressed, uncompressed = fields[4], fields[7], fields[8], fields[9]
        name_size, extra_size, comment_size = fields[10], fields[11], fields[12]
        name = directory[offset + 46 : offset + 46 + name_size].decode("utf-8")
        local_offset = fields[16]
        if (
            compressed == 0xFFFFFFFF
            or uncompressed == 0xFFFFFFFF
            or local_offset == 0xFFFFFFFF
        ):
            extra = directory[offset + 46 + name_size : offset + 46 + name_size + extra_size]
            cursor = 0
            while cursor + 4 <= len(extra):
                tag, size = struct.unpack_from("<HH", extra, cursor)
                body = extra[cursor + 4 : cursor + 4 + size]
                if tag == 1:
                    body_offset = 0
                    if uncompressed == 0xFFFFFFFF:
                        uncompressed = struct.unpack_from("<Q", body, body_offset)[0]
                        body_offset += 8
                    if compressed == 0xFFFFFFFF:
                        compressed = struct.unpack_from("<Q", body, body_offset)[0]
                        body_offset += 8
                    if local_offset == 0xFFFFFFFF:
                        local_offset = struct.unpack_from("<Q", body, body_offset)[0]
                    break
                cursor += 4 + size
        entries[name] = {
            "method": method,
            "crc32": crc,
            "compressed_size": compressed,
            "uncompressed_size": uncompressed,
            "local_header_offset": local_offset,
        }
        offset += 46 + name_size + extra_size + comment_size
    return entries


def choose_sample_points(
    entries: dict[str, dict[str, int]], split: dict[str, Any], *, per_partition: int = 2
) -> list[dict[str, str]]:
    """Choose complete, deterministic RGB/audio/tactile points from train/val only."""
    complete: dict[tuple[str, str], set[str]] = defaultdict(set)
    expression = re.compile(r"^XCapture_data/(\d{5})/(\d{3})/(.+)$")
    for path in entries:
        match = expression.match(path)
        if not match:
            continue
        object_id, point_id, suffix = match.groups()
        for modality in MODALITY_BASENAMES:
            expected_suffix = MODALITY_PATHS[modality]
            if suffix == expected_suffix:
                complete[(object_id, point_id)].add(modality)

    points_by_partition: dict[str, dict[str, list[str]]] = {}
    for partition in ("train", "validation"):
        allowed = set(split["partitions"][partition])
        options: dict[str, list[str]] = defaultdict(list)
        for (object_id, point_id), modalities in complete.items():
            if object_id in allowed and modalities == set(MODALITY_BASENAMES):
                options[object_id].append(point_id)
        if len(options) < per_partition:
            raise RuntimeError(f"not enough complete {partition} objects for sample inspection")
        points_by_partition[partition] = {
            key: sorted(value) for key, value in options.items()
        }

    result: list[dict[str, str]] = []
    for partition, objects in points_by_partition.items():
        ranked_objects = sorted(
            objects,
            key=lambda value: hashlib.sha256(
                f"xcapture-media-sample-v1|seed=17|{partition}|{value}".encode()
            ).hexdigest(),
        )
        for object_id in ranked_objects[:per_partition]:
            points = objects[object_id]
            point_id = min(
                points,
                key=lambda value: hashlib.sha256(
                    f"xcapture-media-point-v1|seed=17|{partition}|{object_id}|{value}".encode()
                ).hexdigest(),
            )
            result.append(
                {"partition": partition, "object_id": object_id, "point_id": point_id}
            )
    return result


def _extract_entry(
    session: requests.Session, path: str, entry: dict[str, int]
) -> bytes:
    start = entry["local_header_offset"]
    local = _http_range(session, start, start + 29)
    fields = struct.unpack("<4s5H3I2H", local)
    if fields[0] != b"PK\x03\x04" or fields[3] != entry["method"]:
        raise RuntimeError(f"local ZIP header mismatch for {path}")
    file_name_size, extra_size = fields[9], fields[10]
    payload_start = start + 30 + file_name_size + extra_size
    compressed = _http_range(
        session, payload_start, payload_start + entry["compressed_size"] - 1
    )
    if entry["method"] == 0:
        payload = compressed
    elif entry["method"] == 8:
        payload = zlib.decompress(compressed, -15)
    else:
        raise RuntimeError(f"unsupported ZIP compression method {entry['method']} for {path}")
    if len(payload) != entry["uncompressed_size"]:
        raise RuntimeError(f"uncompressed size mismatch for {path}")
    if zlib.crc32(payload) & 0xFFFFFFFF != entry["crc32"]:
        raise RuntimeError(f"CRC mismatch for {path}")
    return payload


def run_sample_audit(
    source: dict[str, Any],
    split: dict[str, Any],
    output_dir: Path,
    *,
    per_partition: int = 2,
) -> dict[str, Any]:
    if source.get("status") != "ALLOW_TASK_SCOPED_CROSS_SENSORY_RETRIEVAL":
        raise ValueError("source license gate is not cleared")
    if split.get("archive_sha256") != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("object split does not match the pinned archive")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite sample audit directory: {output_dir}")
    evaluation_ids = set(split["partitions"]["evaluation"])
    session = requests.Session()
    directory, entry_count, _ = _central_directory(session)
    entries = _parse_entries(directory)
    if entry_count != 94973:
        raise RuntimeError("archive entry count changed from the pinned audit")
    sample_points = choose_sample_points(entries, split, per_partition=per_partition)
    if any(item["object_id"] in evaluation_ids for item in sample_points):
        raise RuntimeError("evaluation object was selected for media inspection")

    output_dir.mkdir(parents=True)
    files: list[dict[str, Any]] = []
    for sample_index, point in enumerate(sample_points, start=1):
        prefix = f"XCapture_data/{point['object_id']}/{point['point_id']}/"
        sample_paths = {**MODALITY_PATHS, **AUXILIARY_PATHS}
        for modality, relative_path in sample_paths.items():
            source_path = f"{prefix}{relative_path}"
            payload = _extract_entry(session, source_path, entries[source_path])
            suffix = Path(relative_path).suffix
            output_name = f"sample-{sample_index:02d}-{modality}{suffix}"
            (output_dir / output_name).write_bytes(payload)
            files.append(
                {
                    "sample_id": sample_index,
                    "partition": point["partition"],
                    "object_id": point["object_id"],
                    "point_id": point["point_id"],
                    "modality": modality,
                    "source_path": source_path,
                    "output_path": output_name,
                    "bytes": len(payload),
                    "sha256": _sha256(payload),
                }
            )
    receipt = {
        "dataset_revision": REVISION,
        "archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "central_directory_sha256": EXPECTED_CENTRAL_SHA256,
        "media_policy": (
            f"{len(sample_points)} complete points from train/validation only; "
            "evaluation media unopened"
        ),
        "sample_points": sample_points,
        "files": files,
    }
    (output_dir / "sample-audit.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-report", type=Path, default=SOURCE_REPORT)
    parser.add_argument("--split-manifest", type=Path, default=SPLIT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-partition", type=int, default=2)
    args = parser.parse_args()
    source = json.loads(args.source_report.read_text(encoding="utf-8"))
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    receipt = run_sample_audit(
        source, split, args.output_dir, per_partition=args.per_partition
    )
    summary = {"output_dir": str(args.output_dir), "files": len(receipt["files"])}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
