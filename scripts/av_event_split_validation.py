"""CPU-only validation for frozen AV event manifest identities and splits."""

from __future__ import annotations

from typing import Any

SPLITS = {"fit", "selection", "calibration", "development_validation"}


def validate_video_splits(rows: list[dict[str, Any]]) -> None:
    """Reject duplicate event IDs and any video/media identity crossing splits."""
    seen_ids: set[str] = set()
    video_splits: dict[str, str] = {}
    media_splits: dict[str, str] = {}
    for row in rows:
        row_id, video_id, split = row.get("id"), row.get("video_id"), row.get("split")
        if not isinstance(row_id, str) or not row_id or row_id in seen_ids:
            raise ValueError("event IDs must be nonempty and unique")
        if not isinstance(video_id, str) or not video_id or split not in SPLITS:
            raise ValueError("each row needs a video ID and recognized split")
        seen_ids.add(row_id)
        previous = video_splits.setdefault(video_id, split)
        if previous != split:
            raise ValueError("video appears in multiple splits")
        media_hash = row.get("media_sha256")
        if media_hash is not None:
            if (
                not isinstance(media_hash, str)
                or len(media_hash) != 64
                or any(char not in "0123456789abcdef" for char in media_hash)
            ):
                raise ValueError("media_sha256 must be a lowercase SHA-256 digest")
            previous_media_split = media_splits.setdefault(media_hash, split)
            if previous_media_split != split:
                raise ValueError("media identity appears in multiple splits")
