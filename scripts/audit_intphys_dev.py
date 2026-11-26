"""Validate the released IntPhys development archive without extracting media."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

_VIDEO_PATH = re.compile(
    r"^dev/(?P<block>O[123])/(?P<scene>\d+)/(?P<rendition>\d+)/"
    r"(?P<kind>status\.json|scene/scene_(?P<scene_frame>\d+)\.png|"
    r"depth/depth_(?P<depth_frame>\d+)\.png|"
    r"masks/masks_(?P<mask_frame>\d+)\.png)$"
)
_MODALITIES = ("scene", "depth", "masks")
_FRAME_GROUP = {"scene": "scene_frame", "depth": "depth_frame", "masks": "mask_frame"}


def audit_archive(
    archive_path: Path,
    *,
    expected_md5: str | None = None,
    expected_frames_per_video: int = 100,
) -> dict[str, Any]:
    """Check archive layout, frame completeness, and matched label groups."""
    if expected_frames_per_video < 1:
        raise ValueError("expected_frames_per_video must be positive")

    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with archive_path.open("rb") as archive_file:
        for chunk in iter(lambda: archive_file.read(1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    actual_md5 = md5.hexdigest()
    if expected_md5 and actual_md5.lower() != expected_md5.lower():
        raise ValueError(f"MD5 mismatch: expected {expected_md5.lower()}, got {actual_md5}")

    videos: dict[tuple[str, str, str], dict[str, Any]] = {}
    member_count = 0
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            member_count += 1
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe archive path: {member.name}")
            if not member.isfile():
                continue
            match = _VIDEO_PATH.fullmatch(member.name)
            if not match:
                raise ValueError(f"unexpected IntPhys dev member: {member.name}")
            groups = match.groupdict()
            block = groups["block"]
            scene = groups["scene"]
            rendition = groups["rendition"]
            kind = groups["kind"]
            key = (block, scene, rendition)
            entry = videos.setdefault(
                key,
                {
                    "label": None,
                    "frames": {name: set() for name in _MODALITIES},
                    "scene_hashes": {},
                },
            )
            if kind == "status.json":
                if entry["label"] is not None:
                    raise ValueError(f"duplicate status metadata for {key}")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"cannot read status metadata for {key}")
                status = json.load(extracted)
                label = status.get("header", {}).get("is_possible")
                if not isinstance(label, bool):
                    raise ValueError(f"missing boolean is_possible for {key}")
                entry["label"] = label
                continue

            modality = kind.split("/", maxsplit=1)[0]
            frame_text = groups[_FRAME_GROUP[modality]]
            frame_number = int(frame_text)
            if frame_number in entry["frames"][modality]:
                raise ValueError(f"duplicate {modality} frame {frame_number} for {key}")
            entry["frames"][modality].add(frame_number)
            if modality == "scene":
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"cannot read scene frame {frame_number} for {key}")
                entry["scene_hashes"][frame_number] = hashlib.sha256(extracted.read()).hexdigest()

    if not videos:
        raise ValueError("archive contains no IntPhys dev videos")

    expected_frames = set(range(1, expected_frames_per_video + 1))
    by_block: dict[str, Counter[str]] = defaultdict(Counter)
    groups: dict[tuple[str, str], list[bool]] = defaultdict(list)
    entries_by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (block, scene, _rendition), entry in videos.items():
        if entry["label"] is None:
            raise ValueError(f"missing status metadata for {(block, scene)}")
        for modality, frame_numbers in entry["frames"].items():
            if frame_numbers != expected_frames:
                raise ValueError(
                    f"incomplete {modality} sequence for {(block, scene)}: "
                    f"expected {expected_frames_per_video} frames, got "
                    f"{len(frame_numbers)}"
                )
        by_block[block]["videos"] += 1
        by_block[block]["possible" if entry["label"] else "impossible"] += 1
        groups[(block, scene)].append(entry["label"])
        entries_by_group[(block, scene)].append(entry)

    cross_label_pair_same_index_matches: list[int] = []
    cross_label_pair_shared_frame_counts: list[int] = []
    for group, labels in groups.items():
        if len(labels) != 4 or labels.count(True) != 2 or labels.count(False) != 2:
            raise ValueError(
                f"matched scene group {group} must have 4 renditions with "
                f"2 possible and 2 impossible; got {labels}"
            )
        possible = [entry for entry in entries_by_group[group] if entry["label"]]
        impossible = [entry for entry in entries_by_group[group] if not entry["label"]]
        for possible_entry, impossible_entry in itertools.product(possible, impossible):
            possible_hashes = possible_entry["scene_hashes"]
            impossible_hashes = impossible_entry["scene_hashes"]
            cross_label_pair_same_index_matches.append(
                sum(possible_hashes[index] == impossible_hashes[index] for index in expected_frames)
            )
            cross_label_pair_shared_frame_counts.append(
                sum(
                    (
                        Counter(possible_hashes.values()) & Counter(impossible_hashes.values())
                    ).values()
                )
            )

    return {
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_md5": actual_md5,
        "archive_sha256": sha256.hexdigest(),
        "archive_members": member_count,
        "video_count": len(videos),
        "scene_group_count": len(groups),
        "frames_per_video_per_modality": expected_frames_per_video,
        "modalities": sorted(_MODALITIES),
        "blocks": {name: dict(sorted(counts.items())) for name, counts in sorted(by_block.items())},
        "matched_groups_have_two_labels_each": True,
        "cross_label_scene_frame_overlap": {
            "video_pairs": len(cross_label_pair_same_index_matches),
            "pairs_with_any_shared_frame": sum(
                count > 0 for count in cross_label_pair_shared_frame_counts
            ),
            "same_index_equal_frame_total": sum(cross_label_pair_same_index_matches),
            "same_index_equal_frame_fraction": sum(cross_label_pair_same_index_matches)
            / (len(cross_label_pair_same_index_matches) * expected_frames_per_video),
            "shared_frame_content_total": sum(cross_label_pair_shared_frame_counts),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--expected-md5")
    parser.add_argument("--expected-frames-per-video", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_archive(
        args.archive,
        expected_md5=args.expected_md5,
        expected_frames_per_video=args.expected_frames_per_video,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
