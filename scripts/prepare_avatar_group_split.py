"""Create an AVATAR source-video group split from ZIP names only."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

ARCHIVE_BYTES = 3_785_157_586
ARCHIVE_SHA256 = "5ce5be7fa6dda85889750f6c7cc2fad807527063e7d7090c664170a66b078a68"
REVISION = "a7f0146ff704799d044133e61137836cacc1c233"
SEED = 17
NAME = re.compile(r"video/([A-Za-z0-9_-]{11})_(\d+)\.mp4\Z")


def _source_video_id(member: str) -> str | None:
    if not member.startswith("video/") or not member.lower().endswith(".mp4"):
        return None
    match = NAME.fullmatch(member)
    if match is None:
        raise ValueError(f"unrecognized AVATAR video member name: {member}")
    return match.group(1)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids_sha256(ids: list[str]) -> str:
    return _sha256(("\n".join(sorted(ids)) + "\n").encode("utf-8"))


def build_manifest(archive: Path, *, check_pin: bool = True) -> dict:
    archive = Path(archive)
    actual_bytes = archive.stat().st_size
    actual_hash = _file_sha256(archive)
    if check_pin and (actual_bytes != ARCHIVE_BYTES or actual_hash != ARCHIVE_SHA256):
        raise ValueError("AVATAR video archive does not match pinned revision")

    with zipfile.ZipFile(archive) as bundle:
        members = [entry.filename for entry in bundle.infolist()]
    video_members = [name for name in members if name.startswith("video/")]
    videos = [name for name in video_members if name.lower().endswith(".mp4")]
    if len(videos) != len(set(videos)):
        raise ValueError("duplicate video archive member names")
    counts = Counter(_source_video_id(name) for name in videos)
    if not videos or None in counts:
        raise ValueError("no valid AVATAR video members found")

    ids = sorted(counts)
    ordered = sorted(
        ids,
        key=lambda source_id: _sha256(f"avatar-source-group-v1:{SEED}:{source_id}".encode()),
    )
    first = len(ordered) * 8 // 10
    second = len(ordered) * 9 // 10
    splits = {
        "train": sorted(ordered[:first]),
        "validation": sorted(ordered[first:second]),
        "audit": sorted(ordered[second:]),
    }
    if set(splits["train"]) & set(splits["validation"]):
        raise ValueError("train/validation source-ID overlap")
    if set(splits["train"]) & set(splits["audit"]):
        raise ValueError("train/audit source-ID overlap")
    if set(splits["validation"]) & set(splits["audit"]):
        raise ValueError("validation/audit source-ID overlap")

    manifest = {
        "format": "avatar-source-video-group-split-v1",
        "dataset_revision": REVISION,
        "video_archive_bytes": actual_bytes,
        "video_archive_sha256": actual_hash,
        "split_seed": SEED,
        "split_policy": "sha256 rank by source YouTube ID; floor(0.8N), floor(0.9N) boundaries",
        "ignored_mac_osx_sidecars": sum(name.startswith("__MACOSX/") for name in members),
        "video_clip_count": len(videos),
        "source_video_group_count": len(ids),
        "groups_with_multiple_clips": sum(count > 1 for count in counts.values()),
        "maximum_clips_per_group": max(counts.values()),
        "splits": {
            name: {
                "source_video_group_count": len(group_ids),
                "video_clip_count": sum(counts[source_id] for source_id in group_ids),
                "source_video_ids": group_ids,
                "source_video_ids_sha256": _ids_sha256(group_ids),
            }
            for name, group_ids in splits.items()
        },
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["manifest_sha256"] = _sha256(canonical)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = build_manifest(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "splits"}, sort_keys=True))


if __name__ == "__main__":
    main()
