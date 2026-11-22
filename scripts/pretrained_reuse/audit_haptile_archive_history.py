"""Audit pinned HapTile archive-history metadata without downloading media."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DATASET = "HapTile2026/HapTile"
OLD_REVISION = "917407abe013ac6a4c69544832900eeb2e879b28"
CURRENT_REVISION = "54887384f1fe9e93b799e104194a582f9d8840b8"
RANGE_BYTES = 65_536
OLD_PATH = "Data/remove_suger_bag.zip"
CURRENT_PATH = "Data/remove_sugar_bag.zip"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def parse_zip_tail(tail: bytes, *, absolute_offset: int, archive_size: int) -> list[str]:
    """Return central-directory names from a bounded ZIP tail range."""
    eocd_signature = b"PK\x05\x06"
    central_signature = b"PK\x01\x02"
    eocd = tail.rfind(eocd_signature)
    if eocd < 0 or eocd + 22 > len(tail):
        raise ValueError("ZIP end-of-central-directory record missing or truncated")
    comment_length = int.from_bytes(tail[eocd + 20 : eocd + 22], "little")
    if eocd + 22 + comment_length != len(tail):
        raise ValueError("ZIP tail does not end at the declared archive boundary")
    disk_number = int.from_bytes(tail[eocd + 4 : eocd + 6], "little")
    central_disk = int.from_bytes(tail[eocd + 6 : eocd + 8], "little")
    disk_entries = int.from_bytes(tail[eocd + 8 : eocd + 10], "little")
    entry_count = int.from_bytes(tail[eocd + 10 : eocd + 12], "little")
    central_size = int.from_bytes(tail[eocd + 12 : eocd + 16], "little")
    central_offset = int.from_bytes(tail[eocd + 16 : eocd + 20], "little")
    if disk_number or central_disk or disk_entries != entry_count:
        raise ValueError("multi-disk ZIP archives are not supported")
    if 0xFFFF in (disk_entries, entry_count) or 0xFFFFFFFF in (central_size, central_offset):
        raise ValueError("ZIP64 archive requires a larger parser")
    if central_offset < absolute_offset or central_offset + central_size > archive_size:
        raise ValueError("central directory is not contained in the fetched tail")

    names: list[str] = []
    position = central_offset - absolute_offset
    for _ in range(entry_count):
        if tail[position : position + 4] != central_signature:
            raise ValueError(f"invalid central-directory entry at offset {position}")
        flags = int.from_bytes(tail[position + 8 : position + 10], "little")
        name_length = int.from_bytes(tail[position + 28 : position + 30], "little")
        extra_length = int.from_bytes(tail[position + 30 : position + 32], "little")
        entry_comment_length = int.from_bytes(tail[position + 32 : position + 34], "little")
        start = position + 46
        end = start + name_length
        if end + extra_length + entry_comment_length > len(tail):
            raise ValueError("central-directory entry is truncated")
        encoding = "utf-8" if flags & 0x800 else "cp437"
        names.append(tail[start:end].decode(encoding))
        position = end + extra_length + entry_comment_length
    if position != central_offset - absolute_offset + central_size:
        raise ValueError("central-directory size does not match parsed entries")
    return names


def request_bytes(url: str, *, byte_range: tuple[int, int] | None = None) -> bytes:
    headers = {"User-Agent": "tiny-omni-decision-source-audit/1.0"}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=45) as response:
        payload = response.read()
        if byte_range is not None:
            expected_range = f"bytes {byte_range[0]}-{byte_range[1]}/"
            content_range = response.headers.get("Content-Range", "")
            if response.status != 206 or not content_range.startswith(expected_range):
                raise RuntimeError(
                    f"expected partial response; got {response.status} {content_range}"
                )
        elif response.status != 200:
            raise RuntimeError(f"expected HTTP 200; got {response.status}")
    return payload


def get_tree(revision: str) -> tuple[bytes, list[dict[str, Any]]]:
    url = (
        f"https://huggingface.co/api/datasets/{DATASET}/tree/{revision}?recursive=true&expand=true"
    )
    raw = request_bytes(url)
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError(f"unexpected tree response for {revision}")
    return raw, parsed


def inspect_archive(revision: str, path: str, metadata: dict[str, Any]) -> dict[str, Any]:
    size = int(metadata["size"])
    if size < RANGE_BYTES:
        raise ValueError(f"archive smaller than required tail range: {path}")
    start = size - RANGE_BYTES
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{path}"
    tail = request_bytes(url, byte_range=(start, size - 1))
    if len(tail) != RANGE_BYTES:
        raise RuntimeError(f"short range response for {path}: {len(tail)} bytes")
    names = parse_zip_tail(tail, absolute_offset=start, archive_size=size)
    trajectory_names = sorted(name for name in names if name.endswith("/trajectory.h5"))
    return {
        "revision": revision,
        "path": path,
        "archive_bytes": size,
        "git_blob_oid": metadata.get("oid"),
        "lfs_oid": metadata.get("lfs", {}).get("oid"),
        "range": {
            "start": start,
            "end": size - 1,
            "bytes": len(tail),
            "sha256": sha256_bytes(tail),
        },
        "central_directory_entries": len(names),
        "trajectory_h5_count": len(trajectory_names),
        "trajectory_h5_paths": trajectory_names,
    }


def build_audit(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing audit output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    commits_url = f"https://huggingface.co/api/datasets/{DATASET}/commits/main?limit=100"
    commits_raw = request_bytes(commits_url)
    commits = json.loads(commits_raw)
    commit_by_id = {row.get("id"): row for row in commits if isinstance(row, dict)}
    if OLD_REVISION not in commit_by_id or CURRENT_REVISION not in commit_by_id:
        raise ValueError("pinned revisions are absent from the public commit listing")

    old_tree_raw, old_tree = get_tree(OLD_REVISION)
    current_tree_raw, current_tree = get_tree(CURRENT_REVISION)
    old_files = {row["path"]: row for row in old_tree if row.get("type") == "file"}
    current_files = {row["path"]: row for row in current_tree if row.get("type") == "file"}
    old_zip_files = {path: row for path, row in old_files.items() if path.lower().endswith(".zip")}
    current_zip_files = {
        path: row for path, row in current_files.items() if path.lower().endswith(".zip")
    }
    if OLD_PATH not in old_files or OLD_PATH in current_files or CURRENT_PATH not in current_files:
        raise ValueError("unexpected spelling-alias archive history")
    if old_files[CURRENT_PATH].get("lfs", {}).get("oid") != current_files[CURRENT_PATH].get(
        "lfs", {}
    ).get("oid"):
        raise ValueError("canonical archive changed between compared revisions")

    old_archive = inspect_archive(OLD_REVISION, OLD_PATH, old_files[OLD_PATH])
    current_archive = inspect_archive(CURRENT_REVISION, CURRENT_PATH, current_files[CURRENT_PATH])
    old_names = set(old_archive["trajectory_h5_paths"])
    current_names = set(current_archive["trajectory_h5_paths"])

    (output_dir / "commits-response.json").write_bytes(commits_raw)
    (output_dir / "old-tree-response.json").write_bytes(old_tree_raw)
    (output_dir / "current-tree-response.json").write_bytes(current_tree_raw)

    return {
        "format": "haptile_archive_history_range_audit_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset": DATASET,
        "method": (
            "Pinned Hugging Face commit/tree metadata plus the final 65,536 bytes of each ZIP; "
            "no archive payload or media blocks downloaded."
        ),
        "api_responses": {
            "commits": {
                "url": commits_url,
                "bytes": len(commits_raw),
                "sha256": sha256_bytes(commits_raw),
            },
            "old_tree": {
                "revision": OLD_REVISION,
                "bytes": len(old_tree_raw),
                "sha256": sha256_bytes(old_tree_raw),
            },
            "current_tree": {
                "revision": CURRENT_REVISION,
                "bytes": len(current_tree_raw),
                "sha256": sha256_bytes(current_tree_raw),
            },
        },
        "commits": {
            revision: {
                "title": commit_by_id[revision].get("title"),
                "created_at": commit_by_id[revision].get("date"),
            }
            for revision in (OLD_REVISION, CURRENT_REVISION)
        },
        "tree_archive_counts": {
            "before_spelling_alias_removal": len(old_zip_files),
            "pinned_source_revision": len(current_zip_files),
            "removed_alias_path": OLD_PATH,
        },
        "tree_zip_comparison": {
            "common_zip_count": len(old_zip_files.keys() & current_zip_files.keys()),
            "common_zip_identical_git_oids": sum(
                old_zip_files[path].get("oid") == current_zip_files[path].get("oid")
                for path in old_zip_files.keys() & current_zip_files.keys()
            ),
            "common_zip_different_git_oids": sorted(
                path
                for path in old_zip_files.keys() & current_zip_files.keys()
                if old_zip_files[path].get("oid") != current_zip_files[path].get("oid")
            ),
            "old_revision_only_zip_paths": sorted(old_zip_files.keys() - current_zip_files.keys()),
            "current_revision_only_zip_paths": sorted(
                current_zip_files.keys() - old_zip_files.keys()
            ),
        },
        "archives": {"removed_alias": old_archive, "canonical": current_archive},
        "name_comparison": {
            "shared_trajectory_paths": len(old_names & current_names),
            "removed_alias_only_trajectory_paths": sorted(old_names - current_names),
            "canonical_only_trajectory_paths": sorted(current_names - old_names),
        },
        "interpretation": {
            "confirmed": (
                "An earlier public revision contained an extra misspelled "
                "remove_suger_bag.zip archive; the later pinned revision removed it "
                "while retaining the canonical remove_sugar_bag.zip with the same "
                "LFS object hash across revisions."
            ),
            "not_proven": (
                "Central-directory name overlap does not prove payload identity. "
                "The 26-record paper-to-pinned-release difference is not reconciled "
                "by this audit; no content blocks were downloaded to inspect HDF5 contents."
            ),
            "source_disposition": (
                "REVIEW_NOT_ADMITTED; development-only task-held-out results remain "
                "limited to the frozen conversion and cannot support unseen-operator "
                "or product-quality claims."
            ),
        },
        "privacy_and_scope": {
            "archive_bytes_fetched": RANGE_BYTES * 2,
            "media_or_hdf5_payload_opened": False,
            "sealed_audit_accessed": False,
            "training_or_model_evaluation": False,
        },
        "output_directory": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="new, nonexistent external output directory"
    )
    args = parser.parse_args()
    result = build_audit(args.output)
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"result_sha256={sha256_bytes(result_path.read_bytes())}")


if __name__ == "__main__":
    main()
