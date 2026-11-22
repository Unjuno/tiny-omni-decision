"""Factorial speech-to-screen palette task construction from Lombard TRAIN only."""

from __future__ import annotations

import hashlib
import itertools
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

COLORS = ("blue", "green", "red", "white")
RGB = {
    "blue": (37, 99, 235),
    "green": (34, 197, 94),
    "red": (239, 68, 68),
    "white": (248, 250, 252),
}
POSITIONS = ((0.25, 0.34), (0.75, 0.34), (0.25, 0.68), (0.75, 0.68))
SPLIT_ORDER = ("train", "selection", "calibration", "evaluation")
SCREEN_PARITY = {"train": 0, "selection": 1, "calibration": 1, "evaluation": 1}
SCREEN_VARIANT_SALT = {"train": 11, "selection": 23, "calibration": 37, "evaluation": 53}


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def permutation_parity(colors: tuple[str, ...]) -> int:
    if len(colors) != len(COLORS) or set(colors) != set(COLORS):
        raise ValueError("layout must contain each palette color exactly once")
    indices = [COLORS.index(color) for color in colors]
    inversions = sum(
        indices[left] > indices[right]
        for left in range(len(indices))
        for right in range(left + 1, len(indices))
    )
    return inversions % 2


def layouts_by_parity() -> dict[int, tuple[tuple[str, ...], ...]]:
    result = {
        parity: tuple(
            permutation
            for permutation in itertools.permutations(COLORS)
            if permutation_parity(permutation) == parity
        )
        for parity in (0, 1)
    }
    for layouts in result.values():
        counts = Counter(
            (color, position)
            for layout in layouts
            for position, color in enumerate(layout)
        )
        if len(layouts) != 12 or set(counts.values()) != {3}:
            raise RuntimeError("permutation parity partition is not color-position balanced")
    return result


def split_speaker_groups(
    rows: list[dict[str, Any]],
    *,
    seed: int = 17,
    counts: tuple[int, int, int, int] = (24, 4, 4, 4),
) -> dict[str, str]:
    """Assign indivisible source speaker/content groups in a fixed hash order."""
    if sum(counts) <= 0 or any(type(value) is not int or value < 1 for value in counts):
        raise ValueError("all four split group counts must be positive integers")
    by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group = row.get("speaker_group")
        color = (row.get("targets") or {}).get("color")
        if not isinstance(group, str) or color not in COLORS:
            raise ValueError("TRAIN rows require a speaker_group and one valid color target")
        by_group[group].add(color)
    if any(colors != set(COLORS) for colors in by_group.values()):
        raise ValueError("every indivisible speaker group must support all four color labels")
    if len(by_group) != sum(counts):
        raise ValueError("fixed split group counts must cover every TRAIN speaker group")

    ordered = sorted(
        by_group,
        key=lambda group: (_digest(f"spoken-palette-v1|{seed}|speaker|{group}"), group),
    )
    assignment: dict[str, str] = {}
    offset = 0
    for split, count in zip(SPLIT_ORDER, counts, strict=True):
        for group in ordered[offset : offset + count]:
            assignment[group] = split
        offset += count
    if len(assignment) != len(by_group):
        raise RuntimeError("speaker group split did not assign every group exactly once")
    return assignment


def render_palette_screen(
    path: Path,
    colors_by_position: tuple[str, ...],
    *,
    seed_text: str,
) -> str:
    """Render four unlabeled color choices and return the PNG byte hash."""
    if len(colors_by_position) != 4 or set(colors_by_position) != set(COLORS):
        raise ValueError("screen needs a permutation of all four colors")
    seed = int(_digest(seed_text)[:16], 16)
    rng = random.Random(seed)
    background = rng.choice(((245, 247, 250), (238, 242, 247), (250, 250, 248), (241, 244, 246)))
    image = Image.new("RGB", (160, 210), background)
    draw = ImageDraw.Draw(image)
    # A quiet panel gives DINO a stable context; no target/query text or labels are drawn.
    panel = (10, 18, 150, 192)
    draw.rounded_rectangle(panel, radius=10, fill=(255, 255, 255), outline=(213, 219, 227), width=2)
    centers = ((40, 72), (120, 72), (40, 143), (120, 143))
    base_size = rng.choice((32, 36, 40))
    for (center_x, center_y), color in zip(centers, colors_by_position, strict=True):
        jitter_x = rng.choice((-2, -1, 0, 1, 2))
        jitter_y = rng.choice((-2, -1, 0, 1, 2))
        half = base_size // 2
        box = (center_x - half + jitter_x, center_y - half + jitter_y,
               center_x + half + jitter_x, center_y + half + jitter_y)
        draw.rounded_rectangle(box, radius=5, fill=RGB[color], outline=(55, 65, 81), width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False, compress_level=9)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_rows_for_split(
    rows: list[dict[str, Any]],
    *,
    split: str,
    speaker_splits: dict[str, str],
    audio_feature_index: dict[str, int],
    output_root: Path,
    variants_per_layout: int = 2,
    seed: int = 17,
) -> list[dict[str, Any]]:
    if split not in SPLIT_ORDER or variants_per_layout < 1:
        raise ValueError("valid split and positive layout variant count required")
    parity = SCREEN_PARITY[split]
    layouts = layouts_by_parity()[parity]
    chosen = [row for row in rows if speaker_splits.get(row.get("speaker_group")) == split]
    if not chosen:
        raise ValueError(f"no TRAIN rows assigned to {split}")
    source_ids = [row.get("source_id") for row in chosen]
    if any(not isinstance(value, str) or not value for value in source_ids):
        raise ValueError("every training row needs a source_id")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("duplicate audio source IDs within split")

    screen_hashes: dict[tuple[str, int], str] = {}
    for layout in layouts:
        layout_key = _digest("|".join(layout))[:16]
        for variant in range(variants_per_layout):
            screen_key = _digest(
                f"spoken-palette-v1|{seed}|{split}|{layout_key}|{variant}|"
                f"salt={SCREEN_VARIANT_SALT[split]}"
            )
            relative_screen = Path("screens") / split / f"{screen_key}.png"
            screen_hashes[(layout_key, variant)] = render_palette_screen(
                output_root / relative_screen,
                layout,
                seed_text=screen_key,
            )

    result: list[dict[str, Any]] = []
    for source in sorted(chosen, key=lambda row: row["source_id"]):
        source_id = source["source_id"]
        group = source["speaker_group"]
        color = source["targets"]["color"]
        if source_id not in audio_feature_index:
            raise ValueError("TRAIN source lacks a verified frozen audio feature index")
        for layout in layouts:
            layout_key = _digest("|".join(layout))[:16]
            for variant in range(variants_per_layout):
                screen_key = _digest(
                    f"spoken-palette-v1|{seed}|{split}|{layout_key}|{variant}|"
                    f"salt={SCREEN_VARIANT_SALT[split]}"
                )
                relative_screen = Path("screens") / split / f"{screen_key}.png"
                pixel_hash = screen_hashes[(layout_key, variant)]
                option_positions = list(range(4))
                order_seed = int(
                    _digest(f"order|{seed}|{source_id}|{layout_key}|{variant}")[:16], 16
                )
                random.Random(order_seed).shuffle(option_positions)
                sample_key = _digest(f"{source_id}|{layout_key}|{variant}|{split}")
                options = [
                    {
                        "option_id": "choice-" + _digest(f"{sample_key}|{position}")[:12],
                        "x_norm": POSITIONS[position][0],
                        "y_norm": POSITIONS[position][1],
                    }
                    for position in option_positions
                ]
                target_position = layout.index(color)
                target_index = option_positions.index(target_position)
                row = {
                    "sample_id": sample_key,
                    "source_id": source_id,
                    "speaker_group": group,
                    "split": split,
                    "task_template": "spoken-color-to-screen-action-v1",
                    "audio_feature_index": int(audio_feature_index[source_id]),
                    "screenshot_path": relative_screen.as_posix(),
                    "screenshot_sha256": pixel_hash,
                    "screenshot_shape": [210, 160, 3],
                    "layout_id": layout_key,
                    "render_variant": variant,
                    "option_order_seed": order_seed,
                    "options": options,
                    "target_index": target_index,
                }
                if "target_color" in row or "transcript" in row or "utterance" in row:
                    raise RuntimeError("query text/target label must not enter model input rows")
                result.append(row)
    if len({row["sample_id"] for row in result}) != len(result):
        raise RuntimeError("generated sample IDs are not unique")
    return result


def factorial_support(
    rows: list[dict[str, Any]], source_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Prove both inputs are necessary by balanced source/layout counterfactuals."""
    color_for_source = {row["source_id"]: row["targets"]["color"] for row in source_rows}
    audio_only = Counter()
    image_only = Counter()
    target_by_source: dict[str, set[int]] = defaultdict(set)
    target_by_layout: dict[str, set[int]] = defaultdict(set)
    target_by_source_layout: dict[tuple[str, str], set[int]] = defaultdict(set)
    color_position = Counter()
    color_counts = Counter(color_for_source[row["source_id"]] for row in source_rows)
    unique_media: set[str] = set()
    for row in rows:
        color = color_for_source[row["source_id"]]
        position_index = row["options"][row["target_index"]]
        position = POSITIONS.index((position_index["x_norm"], position_index["y_norm"]))
        audio_only[(row["source_id"], position)] += 1
        image_only[(row["layout_id"], position)] += 1
        target_by_source[row["source_id"]].add(position)
        target_by_layout[row["layout_id"]].add(position)
        target_by_source_layout[(row["source_id"], row["layout_id"])].add(position)
        color_position[(color, position)] += 1
        unique_media.add(row["screenshot_sha256"])
    each_source_balanced = all(
        len(targets) == 4 and len(set(audio_only[(source, position)] for position in range(4))) == 1
        for source, targets in target_by_source.items()
    )
    layout_distributions = {
        layout: tuple(image_only[(layout, position)] for position in range(4))
        for layout in target_by_layout
    }
    variants_per_source_layout = len(rows) / (len(source_rows) * len(target_by_layout))
    image_only_prior_consistent = all(
        sorted(distribution)
        == sorted(int(color_counts[color] * variants_per_source_layout) for color in COLORS)
        for distribution in layout_distributions.values()
    )
    return {
        "rows": len(rows),
        "unique_audio_sources": len(target_by_source),
        "unique_screens": len(unique_media),
        "audio_only_target_positions_per_source": {
            str(position): min(audio_only[(source, position)] for source in target_by_source)
            for position in range(4)
        },
        "image_only_target_positions_per_screen": {
            str(position): min(image_only[(layout, position)] for layout in target_by_layout)
            for position in range(4)
        },
        "audio_only_exactly_uninformative": each_source_balanced,
        "image_only_no_instance_signal_beyond_color_prior": image_only_prior_consistent,
        "image_only_macro_accuracy_ceiling": 0.25,
        "image_only_sample_accuracy_ceiling": max(color_counts.values()) / len(source_rows),
        "source_counts_by_color": dict(color_counts),
        "color_position_counts": {
            f"{color}:{position}": color_position[(color, position)]
            for color in COLORS
            for position in range(4)
        },
        "both_inputs_determine_unique_action": all(
            len(positions) == 1 for positions in target_by_source_layout.values()
        ),
        "sample_count_consistency": len(rows)
        == sum(
            image_only[(layout, position)]
            for layout in target_by_layout
            for position in range(4)
        ),
    }
