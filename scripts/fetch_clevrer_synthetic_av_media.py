"""Fetch and hash only the frozen protocol's CLEVRER training videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from scripts.create_clevrer_synthetic_av_protocol import DATASET, REVISION, sha256_bytes
from scripts.fetch_clevrer_probe import TRAIN_BYTES, TRAIN_ETAG, TRAIN_URL, RangeFile

PROTOCOL_SHA256 = "82b76e813455061769883b6a46517dbfb3a7a54f7593c1093ab92d8cc261fab7"


def _selected_scenes(protocol: dict[str, Any]) -> list[int]:
    if protocol.get("protocol_id") != "clevrer-synthetic-av-timing-v1":
        raise ValueError("unexpected protocol ID")
    if protocol.get("created_from", {}).get("source_dataset") != DATASET:
        raise ValueError("protocol dataset does not match CLEVRER")
    if protocol.get("created_from", {}).get("source_revision") != REVISION:
        raise ValueError("protocol source revision changed")
    if protocol.get("isolation", {}).get("validation_test_or_sealed_data_opened") is not False:
        raise ValueError("protocol does not preserve validation/test/sealed isolation")
    split = protocol.get("split", {}).get("scene_ids")
    if not isinstance(split, dict) or set(split) != {"train", "validation"}:
        raise ValueError("protocol needs train and validation scene groups")
    if set(split["train"]) & set(split["validation"]):
        raise ValueError("protocol scene groups overlap")
    return sorted(set(split["train"]) | set(split["validation"]))


def fetch_media(protocol_path: Path, output_dir: Path, *, archive_file=None) -> dict[str, Any]:
    protocol_bytes = protocol_path.read_bytes()
    protocol_sha256 = sha256_bytes(protocol_bytes)
    if protocol_sha256 != PROTOCOL_SHA256:
        raise ValueError("protocol SHA-256 differs from the frozen protocol receipt")
    protocol = json.loads(protocol_bytes)
    scene_ids = _selected_scenes(protocol)
    if len(scene_ids) != 405:
        raise ValueError("frozen protocol must contain exactly 405 screened train scenes")
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse or overwrite output directory: {output_dir}")
    estimated_size = len(scene_ids) * 2_103_926
    if shutil.disk_usage(output_dir.parent).free < estimated_size * 2:
        raise OSError("insufficient free disk space for estimated video payload and safety margin")
    output_dir.mkdir(parents=True)
    owned_archive = archive_file is None
    if owned_archive:
        archive_file = RangeFile(TRAIN_URL, TRAIN_BYTES, TRAIN_ETAG)
    media_dir = output_dir / "media"
    media_dir.mkdir()

    expected_names = {f"video_{scene:05d}.mp4" for scene in scene_ids}
    hashes_seen: dict[str, int] = {}
    rows = []
    try:
        with zipfile.ZipFile(archive_file) as archive:
            names_by_basename: dict[str, list[zipfile.ZipInfo]] = {}
            for info in archive.infolist():
                if info.filename.casefold().endswith(".mp4"):
                    names_by_basename.setdefault(PurePosixPath(info.filename).name, []).append(info)
            missing = expected_names - set(names_by_basename)
            duplicate = {
                name: values
                for name, values in names_by_basename.items()
                if name in expected_names and len(values) != 1
            }
            if missing or duplicate:
                raise ValueError(
                    f"archive member mismatch: missing={len(missing)}, duplicate={len(duplicate)}"
                )
            for scene in scene_ids:
                name = f"video_{scene:05d}.mp4"
                matches = names_by_basename[name]
                info = matches[0]
                if PurePosixPath(info.filename).name != name or info.file_size < 16:
                    raise ValueError(f"invalid archive metadata for {name}")
                payload = archive.read(info)  # ZipFile verifies each member's CRC.
                if len(payload) != info.file_size or b"ftyp" not in payload[:16]:
                    raise ValueError(f"invalid MP4 payload for {name}")
                digest = hashlib.sha256(payload).hexdigest()
                if digest in hashes_seen:
                    raise ValueError(
                        f"video content duplicate across scenes {hashes_seen[digest]} and {scene}"
                    )
                hashes_seen[digest] = scene
                path = media_dir / name
                with path.open("xb") as stream:
                    stream.write(payload)
                rows.append(
                    {
                        "scene_index": scene,
                        "split": (
                            "train"
                            if scene in set(protocol["split"]["scene_ids"]["train"])
                            else "validation"
                        ),
                        "archive_member": info.filename,
                        "archive_crc32": f"{info.CRC:08x}",
                        "bytes": len(payload),
                        "sha256": digest,
                        "path": f"media/{name}",
                    }
                )
    finally:
        if owned_archive:
            session = getattr(archive_file, "session", None)
            if session is not None:
                session.close()
            archive_file.close()

    if len(rows) != len(scene_ids):
        raise ValueError("downloaded media count does not match frozen scene IDs")
    media_hashes = {row["split"]: set() for row in rows}
    for row in rows:
        media_hashes[row["split"]].add(row["sha256"])
    if media_hashes["train"] & media_hashes["validation"]:
        raise ValueError("media content overlaps across train and validation")
    with (output_dir / "media-manifest.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    receipt = {
        "status": "complete_train_only_media_fetch",
        "dataset": DATASET,
        "revision": REVISION,
        "license": "CC0-1.0",
        "protocol_sha256": protocol_sha256,
        "archive": {
            "url": TRAIN_URL,
            "size_bytes": TRAIN_BYTES,
            "etag": TRAIN_ETAG,
            "full_archive_sha256": "NOT_COMPUTED_RANGE_READ_ONLY",
            "range_bytes_read": getattr(archive_file, "bytes_read", None),
        },
        "scene_count": len(rows),
        "train_scene_count": sum(row["split"] == "train" for row in rows),
        "validation_scene_count": sum(row["split"] == "validation" for row in rows),
        "scene_overlap": 0,
        "media_sha256_overlap": 0,
        "payload_bytes": sum(row["bytes"] for row in rows),
        "validation_test_or_sealed_data_opened": False,
        "entries": rows,
    }
    with (output_dir / "receipt.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = fetch_media(args.protocol, args.output_dir)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output_dir": str(args.output_dir.resolve()),
                "scene_count": receipt["scene_count"],
                "payload_bytes": receipt["payload_bytes"],
                "range_bytes_read": receipt["archive"]["range_bytes_read"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
