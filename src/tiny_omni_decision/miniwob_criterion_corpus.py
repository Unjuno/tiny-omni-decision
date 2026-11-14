"""Data contract for criterion-conditioned MiniWoB screen decisions."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any

import numpy as np
from PIL import Image

TASK_TEMPLATE = "miniwob/click-color-v1:criterion-conditioned-screen"
OPTION_ORDER_XOR = 0x5A17
CRITERION_COUNT = 4
DEFAULT_MASK_ROWS = 64


def masked_screen(image: Image.Image, *, mask_rows: int = DEFAULT_MASK_ROWS) -> Image.Image:
    """Remove the original instruction and target swatch while preserving geometry."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL image")
    if image.size != (160, 210):
        raise ValueError("MiniWoB screenshots must be 160x210")
    if type(mask_rows) is not int or not 1 <= mask_rows < image.height:
        raise ValueError("mask_rows must be an integer within the image")
    result = image.convert("RGB").copy()
    pixels = np.asarray(result).copy()
    pixels[:mask_rows, :, :] = 255
    return Image.fromarray(pixels, mode="RGB")


def validate_mask_layout(
    *,
    query_rect: list[float],
    swatch_rect: list[float],
    option_rects: list[list[float]],
    mask_rows: int = DEFAULT_MASK_ROWS,
) -> None:
    """Require the mask to cover query/swatch and leave every action visible."""
    if type(mask_rows) is not int or not 1 <= mask_rows < 210:
        raise ValueError("mask_rows must be an integer within the screenshot")
    if len(query_rect) != 4 or len(swatch_rect) != 4 or len(option_rects) != 4:
        raise ValueError("the query, swatch, and four options require bounding boxes")
    rectangles = [query_rect, swatch_rect, *option_rects]
    for rect in rectangles:
        if len(rect) != 4 or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not np.isfinite(value) for value in rect
        ):
            raise ValueError("layout rectangles must contain finite numeric x/y/width/height")
    if any(width <= 0 or height <= 0 for _, _, width, height in rectangles):
        raise ValueError("layout rectangles must have positive area")
    if max(query_rect[1] + query_rect[3], swatch_rect[1] + swatch_rect[3]) > mask_rows:
        raise ValueError("query instruction and target swatch must be fully inside the mask")
    if any(rect[1] < mask_rows for rect in option_rects):
        raise ValueError("mask would cover part of an action option")


def pixel_sha256(image: Image.Image) -> str:
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return hashlib.sha256(pixels.tobytes()).hexdigest()


def criterion_text(color: str) -> str:
    if not isinstance(color, str):
        raise ValueError("option colors must be strings")
    normalized = " ".join(color.strip().lower().replace("_", " ").split())
    if not normalized or not re.fullmatch(r"[a-z][a-z -]*", normalized):
        raise ValueError("option color is not a plain English color name")
    return f"Select the {normalized} button."


def build_group_rows(
    *,
    split: str,
    seed: int,
    screenshot_path: str,
    screenshot_sha256: str,
    screenshot_pixel_sha256: str,
    option_colors: list[str],
    option_positions: list[tuple[float, float]],
) -> list[dict[str, Any]]:
    """Create one question per candidate over the same masked screen group."""
    if split not in {"train", "validation", "sealed_audit"}:
        raise ValueError("split must be train, validation, or sealed_audit")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if len(option_colors) != CRITERION_COUNT or len(option_positions) != CRITERION_COUNT:
        raise ValueError("exactly four option colors and positions are required")
    normalized_colors = [" ".join(color.strip().lower().split()) for color in option_colors]
    has_duplicate_or_empty = (
        any(not color for color in normalized_colors)
        or len(set(normalized_colors)) != CRITERION_COUNT
    )
    if has_duplicate_or_empty:
        raise ValueError("each of the four options must have a unique color")
    for name, digest in (("screenshot_sha256", screenshot_sha256),
                         ("screenshot_pixel_sha256", screenshot_pixel_sha256)):
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{name} must be lowercase SHA-256")
    if not isinstance(screenshot_path, str) or not screenshot_path:
        raise ValueError("screenshot_path must be a nonempty relative path")

    permutation = np.random.default_rng(seed ^ OPTION_ORDER_XOR).permutation(CRITERION_COUNT)
    ordered_options: list[dict[str, Any]] = []
    source_to_output: dict[int, str] = {}
    for output_index, source_index_value in enumerate(permutation.tolist()):
        source_index = int(source_index_value)
        x_norm, y_norm = option_positions[source_index]
        if any(isinstance(value, bool) or not np.isfinite(value) or not 0 <= value <= 1
               for value in (x_norm, y_norm)):
            raise ValueError("option positions must be finite normalized coordinates")
        option_id = f"action-{output_index}"
        ordered_options.append({"option_id": option_id, "x_norm": float(x_norm),
                                "y_norm": float(y_norm)})
        source_to_output[source_index] = option_id

    group_id = f"miniwob-seed-{seed}"
    rows: list[dict[str, Any]] = []
    for source_index, color in enumerate(normalized_colors):
        text = criterion_text(color)
        rows.append({
            "sample_id": hashlib.sha256(f"{group_id}\0{text}".encode()).hexdigest(),
            "group_id": group_id,
            "split": split,
            "seed": seed,
            "task_template": TASK_TEMPLATE,
            "question": text,
            "screenshot_path": screenshot_path,
            "screenshot_sha256": screenshot_sha256,
            "screenshot_pixel_sha256": screenshot_pixel_sha256,
            "screenshot_shape": [210, 160, 3],
            "option_order_seed": seed ^ OPTION_ORDER_XOR,
            "options": ordered_options,
            "target_option_id": source_to_output[source_index],
        })
    validate_group_rows(rows)
    return rows


def validate_group_rows(rows: list[dict[str, Any]]) -> None:
    if len(rows) != CRITERION_COUNT:
        raise ValueError("each screen group must contain exactly four criteria")
    group_ids = {row.get("group_id") for row in rows}
    split_names = {row.get("split") for row in rows}
    seeds = {row.get("seed") for row in rows}
    hashes = {row.get("screenshot_sha256") for row in rows}
    pixel_hashes = {row.get("screenshot_pixel_sha256") for row in rows}
    paths = {row.get("screenshot_path") for row in rows}
    options = [row.get("options") for row in rows]
    if any(len(values) != 1 for values in
           (group_ids, split_names, seeds, hashes, pixel_hashes, paths)):
        raise ValueError("all criterion rows must share the same group, split, screen, and seed")
    if group_ids != {f"miniwob-seed-{next(iter(seeds))}"}:
        raise ValueError("group ID must be derived from the underlying simulator seed")
    if len({repr(value) for value in options}) != 1:
        raise ValueError("candidate identities and positions must be fixed within a screen group")
    if len({row.get("question") for row in rows}) != CRITERION_COUNT:
        raise ValueError("the four criteria must have distinct wording")
    option_ids = [option.get("option_id") for option in options[0]]
    targets = [row.get("target_option_id") for row in rows]
    if len(option_ids) != CRITERION_COUNT or len(set(option_ids)) != CRITERION_COUNT:
        raise ValueError("each screen group must have four unique opaque option IDs")
    if set(targets) != set(option_ids):
        raise ValueError("each candidate must be the target of exactly one criterion")
    for row in rows:
        if set(row["options"][0]) != {"option_id", "x_norm", "y_norm"}:
            raise ValueError("model-visible options must not expose source color labels")


def validate_split_groups(splits: dict[str, list[dict[str, Any]]]) -> None:
    group_to_split: dict[str, str] = {}
    group_to_hash: dict[str, str] = {}
    seen_hashes: dict[str, str] = {}
    seen_sample_ids: set[str] = set()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split_name, rows in splits.items():
        if split_name not in {"train", "validation", "sealed_audit"}:
            raise ValueError(f"unknown split: {split_name}")
        for row in rows:
            group_id = row.get("group_id")
            pixel_hash = row.get("screenshot_pixel_sha256")
            if not isinstance(group_id, str) or not isinstance(pixel_hash, str):
                raise ValueError("every row requires a group ID and screen hash")
            if row.get("split") != split_name:
                raise ValueError("row split does not match its corpus partition")
            prior_split = group_to_split.setdefault(group_id, split_name)
            if prior_split != split_name:
                raise ValueError("a screen group appears in multiple splits")
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id or sample_id in seen_sample_ids:
                raise ValueError("sample IDs must be nonempty and globally unique")
            seen_sample_ids.add(sample_id)
            prior_hash = group_to_hash.setdefault(group_id, pixel_hash)
            if prior_hash != pixel_hash:
                raise ValueError("a screen group has inconsistent pixel identities")
            prior_group = seen_hashes.setdefault(pixel_hash, group_id)
            if prior_group != group_id:
                raise ValueError("identical masked screen pixels appear in multiple groups")
            grouped[group_id].append(row)
    for group_rows in grouped.values():
        validate_group_rows(group_rows)
