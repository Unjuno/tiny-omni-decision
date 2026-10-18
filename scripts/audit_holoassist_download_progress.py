"""Audit a live HoloAssist download journal against its immutable selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path


def _expected_files(selection: dict) -> dict[tuple[str, str], dict]:
    selected = selection.get("selection")
    if not isinstance(selected, dict):
        raise ValueError("selection plan has no selection object")
    expected: dict[tuple[str, str], dict] = {}
    for split in ("train", "validation"):
        videos = selected.get(split, {}).get("videos")
        if not isinstance(videos, list):
            raise ValueError(f"selection plan has no {split} video list")
        for row in videos:
            session_id = str(row.get("session_id", ""))
            if (
                not session_id
                or session_id in {".", ".."}
                or not re.fullmatch(r"[A-Za-z0-9 _-]+", session_id)
            ):
                raise ValueError(f"unsafe selected session ID: {session_id!r}")
            key = (split, session_id)
            if key in expected:
                raise ValueError(f"duplicate selected session: {key}")
            size = row.get("size_bytes")
            offset = row.get("offset_data")
            if not isinstance(size, int) or size <= 0:
                raise ValueError(f"invalid expected media size for {key}")
            if not isinstance(offset, int) or offset < 0:
                raise ValueError(f"invalid archive offset for {key}")
            expected[key] = row
    declared_bytes = selected.get("all_selected_video_bytes")
    actual_bytes = sum(int(row["size_bytes"]) for row in expected.values())
    if declared_bytes is not None and int(declared_bytes) != actual_bytes:
        raise ValueError("selection media byte total is inconsistent")
    train_ids = {session for split, session in expected if split == "train"}
    validation_ids = {session for split, session in expected if split == "validation"}
    if train_ids & validation_ids:
        raise ValueError("train/validation selected session IDs overlap")
    return expected


def audit_download_progress(
    *,
    selection: dict,
    selection_sha256: str,
    journal_bytes: bytes,
    media_root: Path,
) -> dict:
    """Check selected identities and file sizes without rehashing media."""
    expected = _expected_files(selection)
    root = media_root.resolve()
    lines = journal_bytes.splitlines(keepends=True)
    incomplete_tail = bool(lines and not lines[-1].endswith((b"\n", b"\r")))
    if incomplete_tail:
        lines = lines[:-1]

    completed: dict[tuple[str, str], dict] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid complete journal row {line_number}") from error
        if not isinstance(item, dict):
            raise ValueError(f"journal row {line_number} is not an object")
        split = str(item.get("split", ""))
        session_id = str(item.get("session_id", ""))
        key = (split, session_id)
        if key in completed:
            raise ValueError(f"duplicate progress-journal entry: {key}")
        row = expected.get(key)
        if row is None:
            raise ValueError(f"unselected progress-journal entry: {key}")
        if item.get("selection_sha256") != selection_sha256:
            raise ValueError(f"selection hash mismatch for {key}")
        if item.get("archive_etag") != selection.get("archive_etag"):
            raise ValueError(f"archive ETag mismatch for {key}")
        if item.get("size_bytes") != int(row["size_bytes"]):
            raise ValueError(f"planned size mismatch for {key}")
        if item.get("archive_offset") != int(row["offset_data"]):
            raise ValueError(f"archive offset mismatch for {key}")
        digest = item.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"invalid recorded SHA-256 for {key}")
        path = Path(str(item.get("path", "")))
        target = root / split / f"{session_id}.mp4"
        if os.path.normcase(str(path.resolve())) != os.path.normcase(str(target.resolve())):
            raise ValueError(f"unexpected output path for {key}")
        if not path.is_file() or path.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"completed media is missing or has wrong size for {key}")
        completed[key] = item

    partial_keys: set[tuple[str, str]] = set()
    partial_bytes = 0
    for path in root.rglob("*.mp4.part"):
        relative = path.relative_to(root)
        if len(relative.parts) != 2 or relative.parts[0] not in {"train", "validation"}:
            raise ValueError(f"unexpected partial media path: {path}")
        split = relative.parts[0]
        session_id = relative.parts[1].removesuffix(".mp4.part")
        key = (split, session_id)
        row = expected.get(key)
        if row is None:
            raise ValueError(f"unselected partial media: {key}")
        if key in completed or key in partial_keys:
            raise ValueError(f"duplicate or completed partial media: {key}")
        expected_partial = root / split / f"{session_id}.mp4.part"
        if path.resolve() != expected_partial.resolve():
            raise ValueError(f"unexpected partial media path: {path}")
        size = path.stat().st_size
        if size > int(row["size_bytes"]):
            raise ValueError(f"partial media exceeds planned size for {key}")
        partial_keys.add(key)
        partial_bytes += size

    unlogged_on_disk = 0
    for key, row in expected.items():
        if key in completed or key in partial_keys:
            continue
        split, session_id = key
        path = root / split / f"{session_id}.mp4"
        if path.exists():
            if not path.is_file() or path.stat().st_size != int(row["size_bytes"]):
                raise ValueError(f"unlogged media is incomplete or wrong size for {key}")
            unlogged_on_disk += 1

    return {
        "schema_version": 1,
        "status": "PASS_PARTIAL_HOLOASSIST_DOWNLOAD_AUDIT",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "selection_sha256": selection_sha256,
        "archive_etag": selection.get("archive_etag"),
        "expected_files": len(expected),
        "journal_rows": len(completed),
        "unique_journal_rows": len(completed),
        "completed_files": len(completed),
        "unlogged_files": len(expected) - len(completed),
        "completed_bytes": sum(int(item["size_bytes"]) for item in completed.values()),
        "unlogged_on_disk_files": unlogged_on_disk,
        "partial_files": len(partial_keys),
        "partial_bytes": partial_bytes,
        "incomplete_trailing_journal_line_ignored": incomplete_tail,
        "media_hashes_recomputed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    selection_bytes = args.selection.read_bytes()
    report = audit_download_progress(
        selection=json.loads(selection_bytes),
        selection_sha256=hashlib.sha256(selection_bytes).hexdigest(),
        journal_bytes=args.journal.read_bytes(),
        media_root=args.media_root,
    )
    result = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(result, end="")
        return
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(result, encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
