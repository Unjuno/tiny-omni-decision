"""Validate the DCASE 2025 dev-train corpus without opening dev-test labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

from tiny_omni_decision.dcase2025 import (
    load_dcase2025_devtrain_queries,
    parse_dcase2025_clip_member,
)

METADATA_SHA256 = "2915b3e3559ff13a33dedcd2fa83a26828b1a1fd1c6bcc85f8548c034637df2b"
GENERATOR_COMMIT = "08e267e5152d37cb7a828f5f52b256f917609132"
UPSTREAM_STARSS_METADATA_SHA256 = "2fa848545a520b4a9d490459de7f079aa49ae16daf0affc859c016f5146981d6"
FROZEN_STARSS_SPLIT_SHA256 = "209423b79c3bf1b98ce926aabf92a7f1f8262a5a8dea69e619f893fe9e1357b6"
MEDIA_MANIFEST_SHA256 = "aba5da16991e93c9ffcfe78c41855532a52a8b08f4c592399aa16382cac06382"
DEVTRAIN_ROOMS = {
    "sony:room21",
    "sony:room22",
    "tau:room4",
    "tau:room6",
    "tau:room7",
    "tau:room9",
    "tau:room12",
    "tau:room13",
    "tau:room14",
}
DEVTEST_ROOMS = {
    "sony:room23",
    "sony:room24",
    "tau:room2",
    "tau:room8",
    "tau:room10",
    "tau:room15",
    "tau:room16",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preflight(metadata_zip: Path, media_manifest: Path, *, seed: int = 17) -> dict[str, object]:
    """Check pinned DCASE metadata and dev-train media availability and grouping."""
    metadata_sha = sha256_file(metadata_zip)
    if metadata_sha != METADATA_SHA256:
        raise ValueError("DCASE metadata archive SHA-256 differs from the pinned release")

    manifest_sha = sha256_file(media_manifest)
    if manifest_sha != MEDIA_MANIFEST_SHA256:
        raise ValueError("STARSS23 media manifest SHA-256 differs from the frozen acquisition")
    manifest = json.loads(media_manifest.read_text(encoding="utf-8"))
    pairs = manifest.get("pairs")
    if manifest.get("status") != "PASS" or not isinstance(pairs, list):
        raise ValueError("frozen media manifest is incomplete")
    media_by_parent: dict[str, dict[str, object]] = {}
    media_room_by_parent: dict[str, str] = {}
    for pair in pairs:
        parent = pair.get("clip")
        files = pair.get("files")
        room = pair.get("room")
        if (
            not isinstance(parent, str)
            or not isinstance(files, dict)
            or not isinstance(room, str)
            or parent in media_by_parent
        ):
            raise ValueError("media manifest has an invalid or duplicate parent recording")
        media_by_parent[parent] = files
        media_room_by_parent[parent] = room

    missing_media: list[str] = []
    verified_media_file_count = 0
    for parent, files in media_by_parent.items():
        for modality in ("audio", "video"):
            record = files.get(modality)
            if not isinstance(record, dict):
                missing_media.append(f"{parent}: missing {modality} manifest entry")
                continue
            path = Path(str(record.get("path", "")))
            expected_size = record.get("bytes")
            if (
                not path.is_file()
                or not isinstance(expected_size, int)
                or path.stat().st_size != expected_size
            ):
                missing_media.append(f"{parent}: missing/size-mismatched {modality}")
            else:
                verified_media_file_count += 1

    devtest_members: list[str] = []
    devtest_rooms: set[str] = set()
    devtrain_rooms: Counter[str] = Counter()
    with zipfile.ZipFile(metadata_zip) as archive:
        for member in archive.namelist():
            if not member.startswith("metadata_dev/dev-test-"):
                continue
            clip = parse_dcase2025_clip_member(member)
            if clip is not None:
                devtest_members.append(member)
                devtest_rooms.add(clip.room_id)

        queries, inventory = load_dcase2025_devtrain_queries(archive, seed=seed)

        for member in archive.namelist():
            if not member.startswith("metadata_dev/dev-train-"):
                continue
            clip = parse_dcase2025_clip_member(member)
            if clip is None:
                continue
            devtrain_rooms[clip.room_id] += 1
            if clip.parent_recording not in media_by_parent:
                missing_media.append(f"{clip.parent_recording}: absent from media manifest")
            elif media_room_by_parent[clip.parent_recording] != clip.room_id:
                missing_media.append(
                    f"{clip.parent_recording}: DCASE/STARSS room identity mismatch"
                )

    if devtest_rooms & set(devtrain_rooms):
        raise ValueError("DCASE dev-train and dev-test rooms overlap")
    if set(devtrain_rooms) != DEVTRAIN_ROOMS or devtest_rooms != DEVTEST_ROOMS:
        raise ValueError("DCASE room inventory differs from the pinned release")
    if inventory["metadata_clip_files"] != 16214 or len(devtest_members) != 13786:
        raise ValueError("DCASE clip inventory differs from the pinned release")
    if inventory["opened_devtest_annotation_members"]:
        raise ValueError("dev-test label content must remain unopened")

    targets = Counter(str(query.target_onscreen) for query in queries)
    classes = Counter(str(query.event_class) for query in queries)
    room_queries = Counter(query.room_id for query in queries)
    report = {
        "status": "PASS" if not missing_media else "FAIL",
        "dataset_record": "doi:10.5281/zenodo.15559774",
        "metadata_sha256": metadata_sha,
        "official_generator_commit": GENERATOR_COMMIT,
        "upstream_starss_metadata_sha256": UPSTREAM_STARSS_METADATA_SHA256,
        "frozen_starss_split_sha256": FROZEN_STARSS_SPLIT_SHA256,
        "media_manifest_sha256": manifest_sha,
        "seed": seed,
        "devtrain_clip_count": inventory["metadata_clip_files"],
        "devtest_member_count_names_only": len(devtest_members),
        "devtest_label_content_opened": False,
        "devtrain_rooms": dict(sorted(devtrain_rooms.items())),
        "devtest_rooms_names_only": sorted(devtest_rooms),
        "annotation_rows": inventory["annotation_rows"],
        "stable_unambiguous_event_segments": inventory["unambiguous_segments"],
        "ambiguous_event_segments_excluded": inventory["ambiguous_segments"],
        "one_query_per_clip": inventory["selected_queries"],
        "selected_sample_id_order_sha256": inventory["selected_id_order_sha256"],
        "selected_targets": dict(sorted(targets.items())),
        "selected_event_classes": dict(sorted(classes.items())),
        "selected_queries_by_room": dict(sorted(room_queries.items())),
        "unique_parent_recordings": len(media_by_parent),
        "verified_media_file_count": verified_media_file_count,
        "missing_or_mismatched_media": sorted(set(missing_media)),
        "training_protocol": "all official dev-train rooms; fixed budget; no validation selection",
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-zip", type=Path, required=True)
    parser.add_argument("--media-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    report = preflight(args.metadata_zip, args.media_manifest, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
