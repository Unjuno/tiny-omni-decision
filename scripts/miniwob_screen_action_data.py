"""Pure validation helpers for frozen MiniWoB++ screen-action examples.

The collector must query the live browser DOM for task-variant eligibility.
MiniWoB's returned ``dom_elements`` intentionally omits empty visual divs, so
it cannot be used to detect the target-color swatch.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def resolve_screen_target_index(
    *,
    query_child_tag: str,
    query_swatch_visible: bool,
    target_color: str,
    option_colors: Sequence[str],
    swatch_rgb: Sequence[int],
    option_center_rgbs: Sequence[Sequence[int]],
) -> int:
    """Verify the rendered swatch is visible and uniquely matches a choice.

    ``query_child_tag`` and ``query_swatch_visible`` must come from a direct
    browser-side query, not from MiniWoB's pruned observation DOM.
    """
    if query_child_tag != "DIV" or not query_swatch_visible:
        raise ValueError("screen-only task requires the visible target-color swatch variant")
    if not isinstance(target_color, str) or not target_color.strip():
        raise ValueError("target color must be a nonempty browser-side label")
    if len(option_colors) < 2 or len(set(option_colors)) != len(option_colors):
        raise ValueError("option colors must be at least two unique values")
    if option_colors.count(target_color) != 1:
        raise ValueError("target color must match exactly one option label")
    if len(swatch_rgb) != 3 or any(
        type(channel) is not int or not 0 <= channel <= 255 for channel in swatch_rgb
    ):
        raise ValueError("swatch RGB must contain three integer channels in [0, 255]")
    if len(option_center_rgbs) != len(option_colors):
        raise ValueError("each option requires one sampled center pixel")
    for color in option_center_rgbs:
        if len(color) != 3 or any(
            type(channel) is not int or not 0 <= channel <= 255 for channel in color
        ):
            raise ValueError("option RGB values must contain integer channels in [0, 255]")
    matches = [
        index for index, color in enumerate(option_center_rgbs) if tuple(color) == tuple(swatch_rgb)
    ]
    expected = option_colors.index(target_color)
    if matches != [expected]:
        raise ValueError("rendered swatch pixels must match only the labeled correct option")
    return expected


def validate_screen_action_row(row: dict[str, Any]) -> None:
    """Fail closed unless a row contains only the screenshot and typed actions."""
    required = {
        "sample_id",
        "episode_group_id",
        "task_template",
        "seed",
        "screenshot_path",
        "screenshot_sha256",
        "screenshot_shape",
        "option_order_seed",
        "options",
        "target_index",
    }
    if set(row) != required:
        raise ValueError("screen-action row has missing or unexpected fields")
    if any(
        not isinstance(row[key], str) or not row[key].strip()
        for key in ("sample_id", "episode_group_id", "task_template", "screenshot_path")
    ):
        raise ValueError("screen-action identifiers and paths must be nonempty strings")
    if not re.fullmatch(r"[0-9a-f]{64}", row["screenshot_sha256"]):
        raise ValueError("screenshot hash must be lowercase SHA-256 hex")
    shape = row["screenshot_shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or any(type(value) is not int or value <= 0 for value in shape)
    ):
        raise ValueError("screenshot shape must contain three positive integers")
    options = row["options"]
    if not isinstance(options, list) or len(options) < 2:
        raise ValueError("at least two action options are required")
    option_ids: list[str] = []
    for option in options:
        if not isinstance(option, dict) or set(option) != {"option_id", "x_norm", "y_norm"}:
            raise ValueError(
                "model-visible options may contain only an ID and normalized coordinates"
            )
        option_id = option["option_id"]
        if not isinstance(option_id, str) or not option_id.strip():
            raise ValueError("option IDs must be nonempty strings")
        option_ids.append(option_id)
        for key in ("x_norm", "y_norm"):
            value = option[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError("normalized option coordinates must be finite values in [0, 1]")
    if len(set(option_ids)) != len(option_ids):
        raise ValueError("option IDs must be unique")
    if type(row["seed"]) is not int or type(row["option_order_seed"]) is not int:
        raise ValueError("seeds must be integers")
    if type(row["target_index"]) is not int or not 0 <= row["target_index"] < len(options):
        raise ValueError("target index must identify a supplied action")


def write_canonical_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> str:
    """Write LF-terminated canonical JSONL as bytes and return its file SHA-256."""
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()
