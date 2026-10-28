"""Identity and split helpers for the expanded DROID site-held-out task."""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Mapping, Sequence
from typing import Any


def canonical_droid_path(value: str) -> str:
    """Canonicalize a source path without discarding its source-relative identity."""
    if not value or not isinstance(value, str):
        raise ValueError("DROID source paths must be non-empty strings")
    return "/".join(
        part for part in value.replace("\\", "/").casefold().split("/") if part
    )


def stable_droid_id(value: str) -> str:
    return hashlib.sha256(canonical_droid_path(value).encode("utf-8")).hexdigest()


def droid_site_from_path(value: str) -> str:
    canonical = canonical_droid_path(value)
    marker = "/r2d2-data-full/"
    if marker not in f"/{canonical}/":
        raise ValueError("DROID source path lacks the r2d2-data-full root")
    site = f"/{canonical}/".split(marker, 1)[1].split("/", 1)[0]
    if not site:
        raise ValueError("DROID source path has an empty collection site")
    return site


def uniform_step_indices(length: int, count: int) -> list[int]:
    """Match the pinned DROID task's rounded, uniformly spaced source indices."""
    if count < 1 or length < count:
        raise ValueError("input window must cover every requested observation")
    if count == 1:
        return [0]
    return [round(index * (length - 1) / (count - 1)) for index in range(count)]


def select_site_holdout(
    site_counts: Mapping[str, int],
    *,
    target_fraction: float = 0.20,
    minimum_fraction: float = 0.15,
    maximum_fraction: float = 0.25,
    minimum_validation_sites: int = 3,
    maximum_validation_sites: int = 5,
    minimum_training_sites: int = 3,
    salt: str = "droid-r2-site-heldout-v1",
) -> dict[str, Any]:
    """Choose a site set by row counts only; labels are intentionally not accepted."""
    if not site_counts or any(
        not site or not isinstance(count, int) or count <= 0
        for site, count in site_counts.items()
    ):
        raise ValueError("site counts must be non-empty positive integers")
    if not 0 < minimum_fraction <= target_fraction <= maximum_fraction < 1:
        raise ValueError("invalid validation fraction bounds")
    if minimum_validation_sites < 1 or minimum_training_sites < 1:
        raise ValueError("at least one site is required in each split")

    sites = sorted(site_counts)
    total = sum(site_counts.values())
    candidates: list[tuple[float, str, tuple[str, ...], int]] = []
    upper = min(maximum_validation_sites, len(sites) - minimum_training_sites)
    for size in range(minimum_validation_sites, upper + 1):
        for held_out in itertools.combinations(sites, size):
            count = sum(site_counts[site] for site in held_out)
            fraction = count / total
            if minimum_fraction <= fraction <= maximum_fraction:
                joined = "|".join(held_out)
                tie_break = hashlib.sha256(f"{salt}|{joined}".encode()).hexdigest()
                candidates.append(
                    (abs(fraction - target_fraction), tie_break, held_out, count)
                )
    if not candidates:
        raise ValueError("no label-independent site holdout satisfies the row bounds")
    _, tie_break, held_out, validation_count = min(candidates)
    return {
        "validation_sites": list(held_out),
        "training_sites": [site for site in sites if site not in held_out],
        "validation_rows": validation_count,
        "eligible_rows": total,
        "validation_fraction": validation_count / total,
        "tie_break_sha256": tie_break,
    }


def validate_site_split(rows: Sequence[Mapping[str, Any]]) -> None:
    """Reject overlap in episode, recording group, site, or exact frame identity."""
    if not rows:
        raise ValueError("split rows must not be empty")
    by_split: dict[str, dict[str, set[str]]] = {
        "train": {key: set() for key in ("id", "group_id", "site", "frame")},
        "validation": {key: set() for key in ("id", "group_id", "site", "frame")},
    }
    for row in rows:
        split = row.get("split")
        if split not in by_split:
            raise ValueError("rows must belong to train or validation")
        for key in ("id", "group_id", "site"):
            value = row.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"row is missing a non-empty {key}")
            by_split[split][key].add(value)
        frame_ids = row.get("input_frame_sha256")
        if not isinstance(frame_ids, list) or any(
            not isinstance(value, str) or not value for value in frame_ids
        ):
            raise ValueError("rows require exact input frame hashes")
        by_split[split]["frame"].update(frame_ids)
    if not by_split["train"]["id"] or not by_split["validation"]["id"]:
        raise ValueError("train and validation must both contain rows")
    for key in ("id", "group_id", "site", "frame"):
        if by_split["train"][key] & by_split["validation"][key]:
            raise ValueError(f"train/validation {key} overlap")
