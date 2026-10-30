"""Task-corpus helpers for the official DCASE 2025 Stereo SELD release.

The training loader intentionally opens only ``dev-train`` annotation members.
The official room-held-out ``dev-test`` member contents are reserved for a
separate, post-freeze evaluation step.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

_CLIP_MEMBER = re.compile(
    r"^metadata_dev/(dev-(train|test)-(sony|tau))/"
    r"(fold\d+_room\d+_mix\d+)_deg(\d{3})_start(\d{4})\.csv$"
)
_ANNOTATION_FIELDS = {"frame", "class", "source", "azimuth", "distance", "onscreen"}


@dataclass(frozen=True)
class DCASE2025Clip:
    member: str
    split: str
    site: str
    parent_recording: str
    room_id: str
    view_degrees: int
    start_frame: int

    @property
    def clip_id(self) -> str:
        return self.member.rsplit("/", 1)[-1][:-4]


@dataclass(frozen=True)
class DCASE2025Query:
    sample_id: str
    clip_id: str
    parent_recording: str
    room_id: str
    split: str
    view_degrees: int
    start_frame: int
    event_frame: int
    event_class: int
    target_onscreen: int


def parse_dcase2025_clip_member(member: str) -> DCASE2025Clip | None:
    """Parse a DCASE clip path; return ``None`` for directory/non-clip entries."""
    match = _CLIP_MEMBER.fullmatch(member)
    if match is None:
        return None
    _split_folder, split, site, parent, view_degrees, start_frame = match.groups()
    room = re.search(r"_room(\d+)_mix\d+$", parent)
    if room is None:
        raise ValueError(f"invalid DCASE recording identity: {parent}")
    return DCASE2025Clip(
        member=member,
        split=split,
        site=site,
        parent_recording=parent,
        room_id=f"{site}:room{room.group(1)}",
        view_degrees=int(view_degrees),
        start_frame=int(start_frame),
    )


def queries_from_annotations(
    clip: DCASE2025Clip, rows: Iterable[Mapping[str, object]]
) -> tuple[list[DCASE2025Query], dict[str, int]]:
    """Convert active event tracks to unambiguous binary screen-status queries.

    Each query represents the midpoint of a contiguous event segment with a
    stable on/off-screen target. If two sources of the same class are active at
    that midpoint, the question would not identify a unique source, so it is
    counted and excluded.
    """
    tracks: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    active_sources_by_class_frame: dict[tuple[int, int], set[int]] = defaultdict(set)
    annotation_count = 0
    for row in rows:
        if set(row) != _ANNOTATION_FIELDS:
            raise ValueError("unexpected DCASE annotation columns")
        try:
            frame = int(row["frame"])
            event_class = int(row["class"])
            source = int(row["source"])
            int(row["azimuth"])
            int(row["distance"])
            onscreen = int(row["onscreen"])
        except (TypeError, ValueError) as error:
            raise ValueError("DCASE annotations must use integer fields") from error
        if not 0 <= frame < 50 or not 0 <= event_class < 13 or onscreen not in (0, 1):
            raise ValueError("DCASE annotation is outside the supported range")
        tracks[(event_class, source, onscreen)].append(frame)
        active_sources_by_class_frame[(event_class, frame)].add(source)
        annotation_count += 1

    queries: list[DCASE2025Query] = []
    ambiguous = 0
    for (event_class, source, onscreen), frames in sorted(tracks.items()):
        ordered = sorted(set(frames))
        segments: list[list[int]] = []
        segment: list[int] = []
        previous: int | None = None
        for frame in ordered:
            if previous is not None and frame != previous + 1:
                segments.append(segment)
                segment = []
            segment.append(frame)
            previous = frame
        if segment:
            segments.append(segment)

        for frames_in_segment in segments:
            event_frame = frames_in_segment[len(frames_in_segment) // 2]
            if len(active_sources_by_class_frame[(event_class, event_frame)]) != 1:
                ambiguous += 1
                continue
            sample_id = (
                f"{clip.clip_id}|frame={event_frame}|class={event_class}|source={source}"
            )
            queries.append(
                DCASE2025Query(
                    sample_id=sample_id,
                    clip_id=clip.clip_id,
                    parent_recording=clip.parent_recording,
                    room_id=clip.room_id,
                    split=clip.split,
                    view_degrees=clip.view_degrees,
                    start_frame=clip.start_frame,
                    event_frame=event_frame,
                    event_class=event_class,
                    target_onscreen=onscreen,
                )
            )
    return queries, {
        "annotation_rows": annotation_count,
        "unambiguous_segments": len(queries),
        "ambiguous_segments": ambiguous,
    }


def select_one_query_per_clip(
    queries: Iterable[DCASE2025Query], *, seed: int = 17
) -> list[DCASE2025Query]:
    """Choose one event segment per clip by a target-independent stable hash."""
    by_clip: dict[str, list[DCASE2025Query]] = defaultdict(list)
    for query in queries:
        by_clip[query.clip_id].append(query)
    selected = []
    for _clip_id, candidates in by_clip.items():
        chosen = min(
            candidates,
            key=lambda item: hashlib.sha256(
                f"{seed}\0{item.sample_id}".encode()
            ).digest(),
        )
        selected.append(chosen)
    return sorted(selected, key=lambda item: item.sample_id)


def load_dcase2025_devtrain_queries(
    archive: zipfile.ZipFile, *, seed: int = 17
) -> tuple[list[DCASE2025Query], dict[str, object]]:
    """Read dev-train members only and return one query per eligible clip."""
    all_queries: list[DCASE2025Query] = []
    inventory: dict[str, object] = {
        "opened_split": "dev-train",
        "opened_devtest_annotation_members": False,
        "metadata_clip_files": 0,
        "clips_with_unambiguous_queries": 0,
        "annotation_rows": 0,
        "ambiguous_segments": 0,
        "unambiguous_segments": 0,
        "clip_counts_by_room": {},
    }
    room_counts: dict[str, int] = defaultdict(int)
    for member in archive.namelist():
        if not member.startswith("metadata_dev/dev-train-"):
            continue
        clip = parse_dcase2025_clip_member(member)
        if clip is None:
            continue
        inventory["metadata_clip_files"] = int(inventory["metadata_clip_files"]) + 1
        room_counts[clip.room_id] += 1
        with archive.open(member) as stream:
            text = io.TextIOWrapper(stream, encoding="utf-8-sig", newline="")
            rows = list(csv.DictReader(text))
        queries, counts = queries_from_annotations(clip, rows)
        for key in ("annotation_rows", "ambiguous_segments", "unambiguous_segments"):
            inventory[key] = int(inventory[key]) + counts[key]
        if queries:
            inventory["clips_with_unambiguous_queries"] = (
                int(inventory["clips_with_unambiguous_queries"]) + 1
            )
            all_queries.extend(queries)
    selected = select_one_query_per_clip(all_queries, seed=seed)
    inventory["selected_queries"] = len(selected)
    inventory["clip_counts_by_room"] = dict(sorted(room_counts.items()))
    inventory["selected_id_order_sha256"] = hashlib.sha256(
        "".join(f"{item.sample_id}\n" for item in selected).encode("utf-8")
    ).hexdigest()
    return selected, inventory


def assert_dcase2025_room_disjoint(
    train: Iterable[DCASE2025Clip | DCASE2025Query],
    heldout: Iterable[DCASE2025Clip | DCASE2025Query],
) -> None:
    """Reject any room, parent recording or clip identity crossing splits."""
    train_rows = list(train)
    heldout_rows = list(heldout)
    for attribute in ("room_id", "parent_recording", "clip_id"):
        left = {getattr(row, attribute) for row in train_rows}
        right = {getattr(row, attribute) for row in heldout_rows}
        if left & right:
            raise ValueError(f"DCASE train/heldout {attribute} overlap")
