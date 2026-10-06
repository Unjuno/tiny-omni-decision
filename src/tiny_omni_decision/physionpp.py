from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from .schema import DecisionExample

PHYSIONPP_SOURCE = "physionpp/readout"
PHYSIONPP_REVISION = "0c53b983ca138962ba636c0936e735490ad4c326"
PHYSIONPP_READOUT_URL = "https://physion-v2.s3.amazonaws.com/readout_data.zip"
PHYSIONPP_ETAG = '"4cf84f667bcc3eaf991dbcc10dd31177-355"'
PHYSIONPP_LAST_MODIFIED = "Sat, 10 Jun 2023 23:02:20 GMT"

_PLACEHOLDERS = {"", "none", "null"}
_STIMULUS_INDEX = re.compile(r"_(\d{4})$")
_MEDIA_SUFFIXES = {"rgb": "_img.mp4", "segmentation": "_id.mp4", "metadata": ".pkl"}


def _path_parts(row: Mapping[str, Any]) -> tuple[str, str]:
    metadata_path = str(row.get("zip_metadata_path", "")).replace("\\", "/")
    parts = metadata_path.split("/")
    if len(parts) < 4 or parts[0] != "readout_data_v1" or parts[-1] != "metadata.json":
        raise ValueError("Physion++ row needs a readout_data_v1 metadata path")
    scenario = str(row.get("scenario", ""))
    if not scenario or scenario != parts[1]:
        raise ValueError("Physion++ scenario does not match its metadata path")
    return "/".join(parts[:-1]), scenario


def _trial_seed(row: Mapping[str, Any]) -> int:
    seed = row.get("trial_seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Physion++ row needs an integer trial_seed")
    return seed


def _is_placeholder(row: Mapping[str, Any]) -> bool:
    return str(row.get("stimulus_name", "")).strip().casefold() in _PLACEHOLDERS


def _media_members(directory: str, index: int) -> dict[str, str]:
    stem = f"{index:04d}"
    return {kind: f"{directory}/{stem}{suffix}" for kind, suffix in _MEDIA_SUFFIXES.items()}


def _record(
    row: Mapping[str, Any],
    *,
    directory: str,
    scenario: str,
    index: int,
    source_revision: str,
    split: str,
    num_frames: int,
) -> dict[str, Any]:
    label = row.get("does_target_contact_zone")
    if not isinstance(label, bool):
        raise ValueError("Physion++ contact-zone label must be a boolean")
    seed = _trial_seed(row)
    media_path = _media_members(directory, index)["rgb"]
    record_id = f"{directory}#{index:04d}"
    return {
        "id": record_id,
        "split": split,
        "state": "A simulated physical-scene video is supplied.",
        "question": "Did the red object contact the yellow-marked target at any point in the clip?",
        "options": ["No", "Yes"],
        "target": "Yes" if label else "No",
        "source_target": label,
        "task_type": "temporal_descriptive",
        "task_group_id": record_id,
        "source_asset_group_id": f"{scenario}:seed={seed}",
        "media": [
            {
                "kind": "video",
                "uri": f"source-ref://physionpp-readout/{source_revision}/{media_path}",
                "license": "MIT",
                "num_frames": num_frames,
            }
        ],
    }


def build_readout_rows(
    records: Iterable[Mapping[str, Any]],
    archive_members: Iterable[str],
    *,
    source_revision: str = PHYSIONPP_REVISION,
    split: str = "readout-development",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Map readout metadata to generic Decision rows without executing PKLs.

    Placeholder stimulus names are accepted only when their metadata directory has
    a complete, contiguous RGB/segmentation/PKL index matching the metadata row order.
    Named rows use their explicit numeric suffix. Rows without a complete media trio
    are counted and excluded; duplicate media identities fail closed if labels differ.
    """
    members = {str(name).replace("\\", "/") for name in archive_members}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        directory, _ = _path_parts(row)
        grouped[directory].append(row)

    output_by_id: dict[str, dict[str, Any]] = {}
    counters: Counter[str] = Counter()
    source_order: list[str] = []
    for directory, rows in grouped.items():
        scenario = _path_parts(rows[0])[1]
        placeholder_flags = [_is_placeholder(row) for row in rows]
        if any(placeholder_flags) and not all(placeholder_flags):
            raise ValueError("Physion++ metadata directory cannot mix named and placeholder rows")

        if all(placeholder_flags):
            counters["placeholder_directories"] += 1
            observed: dict[str, set[int]] = {kind: set() for kind in _MEDIA_SUFFIXES}
            for member in members:
                if not member.startswith(directory + "/"):
                    continue
                basename = member.rsplit("/", 1)[-1]
                for kind, suffix in _MEDIA_SUFFIXES.items():
                    match = re.fullmatch(r"(\d{4})" + re.escape(suffix), basename)
                    if match:
                        observed[kind].add(int(match.group(1)))
            expected_indices = set(range(len(rows)))
            if any(indices != expected_indices for indices in observed.values()):
                raise ValueError(
                    "Physion++ placeholder mapping requires contiguous RGB, segmentation, "
                    "and PKL indices matching metadata row count"
                )
            counters["placeholder_directories_verified"] += 1
            indexed_rows = enumerate(rows)
        else:
            counters["named_directories"] += 1
            indexed = []
            for row in rows:
                stimulus_name = str(row.get("stimulus_name", ""))
                match = _STIMULUS_INDEX.search(stimulus_name)
                if match is None:
                    raise ValueError("Physion++ named stimulus needs a four-digit media suffix")
                indexed.append((int(match.group(1)), row))
            indexed_rows = iter(indexed)

        for index, row in indexed_rows:
            media = _media_members(directory, index)
            if not all(member in members for member in media.values()):
                counters["unmatched_rows"] += 1
                continue
            frame_count = row.get("num_frames")
            if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 1:
                raise ValueError(f"Physion++ frame count is invalid for {media['rgb']}")
            item = _record(
                row,
                directory=directory,
                scenario=scenario,
                index=index,
                source_revision=source_revision,
                split=split,
                num_frames=frame_count,
            )
            identity = item["id"]
            previous = output_by_id.get(identity)
            if previous is not None:
                if (
                    previous["source_target"] != item["source_target"]
                    or previous["task_group_id"] != item["task_group_id"]
                ):
                    raise ValueError(
                        "duplicate Physion++ media identity has conflicting labels or seed"
                    )
                counters["duplicate_rows"] += 1
                continue
            output_by_id[identity] = item
            source_order.append(identity)
            counters[
                "placeholder_rows_mapped" if _is_placeholder(row) else "named_rows_mapped"
            ] += 1

    output = [output_by_id[record_id] for record_id in source_order]
    if len({row["media"][0]["uri"] for row in output}) != len(output):
        raise ValueError("Physion++ mapped rows contain duplicate media URIs")
    counters["mapped_rows"] = len(output)
    counters["scene_seed_groups"] = len({row["source_asset_group_id"] for row in output})
    counters["record_id_sha256"] = hashlib.sha256(
        "\n".join(sorted(str(row["id"]) for row in output)).encode("utf-8")
    ).hexdigest()
    return output, dict(counters)


def assert_physionpp_train_validation_groups(
    train_rows: Iterable[Mapping[str, Any]], validation_rows: Iterable[Mapping[str, Any]]
) -> None:
    """Reject a scene/seed group or media identity appearing in both partitions."""
    train = list(train_rows)
    validation = list(validation_rows)
    train_groups = {str(row.get("source_asset_group_id", "")) for row in train}
    validation_groups = {str(row.get("source_asset_group_id", "")) for row in validation}
    if "" in train_groups | validation_groups:
        raise ValueError("Physion++ rows need source_asset_group_id for scene/seed integrity")
    overlap = train_groups & validation_groups
    if overlap:
        raise ValueError(f"Physion++ scene/seed groups cross train/validation: {sorted(overlap)}")

    def media_key(row: Mapping[str, Any]) -> str:
        media = row["media"][0]
        return str(media.get("sha256") or media.get("uri") or media.get("path") or "")

    train_media = {media_key(row) for row in train}
    validation_media = {media_key(row) for row in validation}
    if "" in train_media | validation_media:
        raise ValueError("Physion++ examples need an identifiable video media reference")
    if train_media & validation_media:
        raise ValueError("Physion++ media identity overlaps train and validation")


def split_readout_examples(
    examples: list[DecisionExample], *, seed: int = 17, validation_fraction: float = 0.2
) -> tuple[list[DecisionExample], list[DecisionExample], dict[str, Any]]:
    """Deterministically partition complete scenario/seed groups into train and dev."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between zero and one")
    grouped: dict[str, list[DecisionExample]] = defaultdict(list)
    for example in examples:
        if example.source != "physionpp/readout" or not example.source_asset_group_id:
            raise ValueError("Physion++ examples need their source and scenario/seed group IDs")
        grouped[example.source_asset_group_id].append(example)
    if len(grouped) < 2:
        raise ValueError("Physion++ split requires at least two distinct scenario/seed groups")

    ordered_groups = sorted(
        grouped,
        key=lambda group: hashlib.sha256(f"{seed}\0physionpp-readout\0{group}".encode()).digest(),
    )
    validation_count = min(
        len(ordered_groups) - 1,
        max(1, round(len(ordered_groups) * validation_fraction)),
    )
    validation_groups = set(ordered_groups[:validation_count])
    train = [
        example.model_copy(update={"split": "train"})
        for example in examples
        if example.source_asset_group_id not in validation_groups
    ]
    validation = [
        example.model_copy(update={"split": "validation"})
        for example in examples
        if example.source_asset_group_id in validation_groups
    ]
    assert_physionpp_train_validation_groups(
        [example.model_dump(mode="json") for example in train],
        [example.model_dump(mode="json") for example in validation],
    )
    from .dataset import check_train_eval_splits

    integrity = check_train_eval_splits(train, validation)
    return (
        train,
        validation,
        {
            "seed": seed,
            "validation_fraction": validation_fraction,
            "group_count": len(ordered_groups),
            "validation_group_count": validation_count,
            "validation_group_order_sha256": hashlib.sha256(
                "\n".join(ordered_groups[:validation_count]).encode("utf-8")
            ).hexdigest(),
            "integrity": integrity,
        },
    )
