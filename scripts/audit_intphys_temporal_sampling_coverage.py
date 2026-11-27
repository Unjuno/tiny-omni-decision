"""Audit IntPhys dev identity loss under predeclared uniform RGB frame sampling."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import tarfile
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

EXPECTED_BYTES = 3_203_229_667
EXPECTED_MD5 = "6a72715007f2f3b0e9546bfa8d8fc39b"
EXPECTED_SHA256 = "441eebe09f6583a18dbcd9d2b6c6b5667c66bac99e3875622e0768dc22e16a0a"
FRAME_COUNTS = (8, 16, 25, 32, 50, 100)
FRAME_RE = re.compile(r"^scene_(\d+)\.png$")


def uniform_frame_numbers(total: int, requested: int) -> tuple[int, ...]:
    """Return 1-based, endpoint-inclusive, unique uniform sample positions."""
    if total < 1 or requested < 1 or requested > total:
        raise ValueError("requested frames must be between one and total frames")
    if requested == 1:
        return (1,)
    numbers = tuple(
        (index * (total - 1)) // (requested - 1) + 1
        for index in range(requested)
    )
    if len(set(numbers)) != requested or numbers[0] != 1 or numbers[-1] != total:
        raise ValueError("uniform sampling did not produce unique endpoint-inclusive frames")
    return numbers


def summarize_group_collisions(
    group_to_rows: dict[str, list[tuple[bool, str]]],
) -> dict[str, Any]:
    """Count exact same-group possible/impossible identities after hashes are frozen."""
    pair_count = 0
    colliding_groups: list[str] = []
    collision_pairs: list[dict[str, str]] = []
    for group_id, rows in sorted(group_to_rows.items()):
        if len(rows) != 4 or sum(label for label, _ in rows) != 2:
            raise ValueError(f"group {group_id} must contain four rows with a 2/2 label balance")
        possible = sorted(identity for label, identity in rows if label)
        impossible = sorted(identity for label, identity in rows if not label)
        matches = [(p, i) for p, i in itertools.product(possible, impossible) if p == i]
        pair_count += len(possible) * len(impossible)
        if matches:
            colliding_groups.append(group_id)
            collision_pairs.extend(
                {"group_id": group_id, "possible_sequence_sha256": p,
                 "impossible_sequence_sha256": i}
                for p, i in matches
            )
    return {
        "scene_groups": len(group_to_rows),
        "cross_label_pairs": pair_count,
        "groups_with_exact_cross_label_sequence_collision": len(colliding_groups),
        "exact_cross_label_sequence_collision_pairs": len(collision_pairs),
        "colliding_group_ids": colliding_groups,
        "collision_pair_hashes": collision_pairs,
    }


def hash_file(path: Path) -> tuple[str, str, int]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            size += len(chunk)
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest(), size


def audit_frame_sampling(path: Path) -> dict[str, Any]:
    """Hash complete ordered RGB source frames, then join labels for collision counts."""
    started_at_utc = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    md5, sha256, size = hash_file(path)
    if (md5, sha256, size) != (EXPECTED_MD5, EXPECTED_SHA256, EXPECTED_BYTES):
        raise ValueError("IntPhys dev archive does not match the pinned, previously audited bytes")

    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile():
                if not member.isdir():
                    raise ValueError(f"unsupported archive member type: {member.name}")
                continue
            parts = PurePosixPath(member.name).parts
            is_status = (
                len(parts) == 5
                and parts[0] == "dev"
                and parts[1] in {"O1", "O2", "O3"}
                and parts[4] == "status.json"
            )
            if is_status:
                key = (parts[1], parts[2], parts[3])
                row = records.setdefault(key, {"frames": {}, "label": None})
                if row["label"] is not None:
                    raise ValueError(f"duplicate status metadata for {key}")
                payload = archive.extractfile(member)
                if payload is None:
                    raise ValueError(f"could not read status metadata for {key}")
                label = json.load(payload).get("header", {}).get("is_possible")
                if not isinstance(label, bool):
                    raise ValueError(f"invalid possible/impossible label for {key}")
                row["label"] = label
                continue
            if (
                len(parts) != 6 or parts[0] != "dev" or parts[1] not in {"O1", "O2", "O3"}
                or parts[4] != "scene"
            ):
                continue
            match = FRAME_RE.fullmatch(parts[5])
            if match is None:
                raise ValueError(f"unexpected RGB frame filename: {member.name}")
            frame_number = int(match.group(1))
            if not 1 <= frame_number <= 100:
                raise ValueError(f"frame number outside 1..100: {member.name}")
            key = (parts[1], parts[2], parts[3])
            row = records.setdefault(key, {"frames": {}, "label": None})
            if frame_number in row["frames"]:
                raise ValueError(f"duplicate RGB frame {frame_number} for {key}")
            payload = archive.extractfile(member)
            if payload is None:
                raise ValueError(f"could not read RGB frame {member.name}")
            row["frames"][frame_number] = hashlib.sha256(payload.read()).hexdigest()

    if len(records) != 360:
        raise ValueError(f"expected 360 matched videos, found {len(records)}")
    groups: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for key, row in records.items():
        if row["label"] is None or set(row["frames"]) != set(range(1, 101)):
            raise ValueError(f"incomplete metadata or RGB frame sequence for {key}")
        groups[key[:2]].append((key[2], row))
    if len(groups) != 90 or any(len(rows) != 4 for rows in groups.values()):
        raise ValueError("expected 90 scene groups with four renditions each")

    policies: list[dict[str, Any]] = []
    for count in FRAME_COUNTS:
        positions = uniform_frame_numbers(100, count)
        group_signatures: dict[str, list[tuple[bool, str]]] = defaultdict(list)
        ordered_hashes = hashlib.sha256()
        for group_key in sorted(groups):
            group_id = f"{group_key[0]}/{group_key[1]}"
            for rendition, row in sorted(groups[group_key]):
                signature_input = "\n".join(row["frames"][number] for number in positions)
                sequence_hash = hashlib.sha256(signature_input.encode("ascii")).hexdigest()
                ordered_hashes.update(f"{group_id}/{rendition}\t{sequence_hash}\n".encode())
                group_signatures[group_id].append((row["label"], sequence_hash))
        summary = summarize_group_collisions(group_signatures)
        policies.append({
            "requested_frame_count": count,
            "selected_frame_numbers_1_based": list(positions),
            "sequence_identity_policy": (
                "SHA-256 over ordered hashes of pinned PNG payloads, in temporal order"
            ),
            "ordered_sample_identity_sha256": ordered_hashes.hexdigest(),
            **summary,
        })
    return {
        "status": "complete_label-blind-frame-identity-first_temporal-coverage-audit",
        "started_at_utc": started_at_utc,
        "ended_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "source": (
            "previously audited official IntPhys dev archive; labels are not blind"
        ),
        "archive_bytes": size,
        "archive_md5": md5,
        "archive_sha256": sha256,
        "videos": len(records),
        "scene_groups": len(groups),
        "frames_per_video": 100,
        "frame_counts_frozen_before_run": list(FRAME_COUNTS),
        "model_or_feature_evaluation_performed": False,
        "training_or_calibration_performed": False,
        "hidden_test_accessed": False,
        "limitations": [
            "identity uses PNG payload hashes, not decoded/resized VideoPrism tensor bytes; "
            "exact collisions are sufficient but not necessary evidence of identical model input",
            "the dev set was previously observed and is not a fresh or sealed evaluation",
            "fewer collisions would not establish that additional frames contain task-relevant "
            "signal or improve decision quality",
        ],
        "sampling_policies": policies,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    result = audit_frame_sampling(args.archive.resolve(strict=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
