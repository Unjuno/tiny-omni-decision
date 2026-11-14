import hashlib
from copy import deepcopy

import numpy as np
import pytest
from PIL import Image, ImageDraw

from tiny_omni_decision.miniwob_criterion_corpus import (
    build_group_rows,
    masked_screen,
    pixel_sha256,
    validate_group_rows,
    validate_mask_layout,
    validate_split_groups,
)


def _screen() -> Image.Image:
    image = Image.new("RGB", (160, 210), "white")
    draw = ImageDraw.Draw(image)
    for x, color in zip((15, 50, 85, 120), ("red", "blue", "green", "yellow"), strict=True):
        draw.rectangle((x, 80, x + 22, 110), fill=color)
    draw.text((5, 8), "Select the target", fill="black")
    draw.rectangle((130, 5, 150, 25), fill="red")
    return image


def _rows(*, split: str = "train", seed: int = 123) -> list[dict]:
    image = masked_screen(_screen())
    pixel_hash = pixel_sha256(image)
    file_hash = hashlib.sha256(pixel_hash.encode()).hexdigest()
    return build_group_rows(
        split=split,
        seed=seed,
        screenshot_path=f"{split}/seed-{seed}.png",
        screenshot_sha256=file_hash,
        screenshot_pixel_sha256=pixel_hash,
        option_colors=["red", "blue", "green", "yellow"],
        option_positions=[(0.16, 0.45), (0.38, 0.45), (0.60, 0.45), (0.82, 0.45)],
    )


def test_mask_removes_top_instruction_and_target_but_keeps_options() -> None:
    original = np.asarray(_screen())
    masked = np.asarray(masked_screen(_screen(), mask_rows=64))
    assert np.all(masked[:64] == 255)
    assert np.array_equal(masked[64:], original[64:])
    assert np.any(masked[80:111] != 255)


def test_same_screen_has_four_criteria_and_one_target_per_candidate() -> None:
    rows = _rows()
    validate_group_rows(rows)
    assert len({row["group_id"] for row in rows}) == 1
    pixel_hash = pixel_sha256(masked_screen(_screen()))
    file_hash = hashlib.sha256(pixel_hash.encode()).hexdigest()
    assert {row["screenshot_sha256"] for row in rows} == {file_hash}
    assert {row["screenshot_pixel_sha256"] for row in rows} == {pixel_hash}
    assert {row["question"] for row in rows} == {
        "Select the red button.",
        "Select the blue button.",
        "Select the green button.",
        "Select the yellow button.",
    }
    assert {row["target_option_id"] for row in rows} == {
        option["option_id"] for option in rows[0]["options"]
    }
    assert all(set(option) == {"option_id", "x_norm", "y_norm"}
               for option in rows[0]["options"])
    assert all("color" not in option for row in rows for option in row["options"])


def test_candidate_order_is_seeded_and_independent_of_criterion_iteration() -> None:
    first = _rows()
    second = _rows()
    assert first == second
    assert len({tuple(option["option_id"] for option in row["options"])
                for row in first}) == 1


def test_duplicate_colors_are_rejected_because_target_would_be_ambiguous() -> None:
    image = masked_screen(_screen())
    with pytest.raises(ValueError, match="unique color"):
        build_group_rows(
            split="train",
            seed=123,
            screenshot_path="train/seed-123.png",
            screenshot_sha256=hashlib.sha256(b"image-file").hexdigest(),
            screenshot_pixel_sha256=pixel_sha256(image),
            option_colors=["red", "blue", "red", "yellow"],
            option_positions=[(0.1, 0.4), (0.3, 0.4), (0.6, 0.4), (0.8, 0.4)],
        )


def test_mask_must_cover_query_and_swatch_without_hiding_any_option() -> None:
    validate_mask_layout(
        query_rect=[0, 0, 160, 58],
        swatch_rect=[130, 5, 20, 20],
        option_rects=[[10 + index * 35, 80, 22, 30] for index in range(4)],
        mask_rows=64,
    )
    with pytest.raises(ValueError, match="target swatch"):
        validate_mask_layout(
            query_rect=[0, 0, 160, 58],
            swatch_rect=[130, 50, 20, 20],
            option_rects=[[10 + index * 35, 80, 22, 30] for index in range(4)],
            mask_rows=64,
        )
    with pytest.raises(ValueError, match="cover part of an action"):
        validate_mask_layout(
            query_rect=[0, 0, 160, 58],
            swatch_rect=[130, 5, 20, 20],
            option_rects=[[10, 60, 22, 30], [45, 80, 22, 30],
                          [80, 80, 22, 30], [115, 80, 22, 30]],
            mask_rows=64,
        )


def test_group_and_identical_pixel_splits_are_rejected() -> None:
    train = _rows()
    with pytest.raises(ValueError, match="multiple splits"):
        validate_split_groups({"train": train, "validation": [dict(train[0], split="validation")]})

    other = _rows(split="validation", seed=456)
    for row in other:
        row["screenshot_pixel_sha256"] = train[0]["screenshot_pixel_sha256"]
    with pytest.raises(ValueError, match="identical masked screen pixels"):
        validate_split_groups({"train": train, "validation": other})


def test_missing_criterion_variant_is_rejected() -> None:
    rows = _rows()
    with pytest.raises(ValueError, match="exactly four criteria"):
        validate_group_rows(rows[:3])


def test_validation_does_not_mutate_rows() -> None:
    rows = _rows()
    original = deepcopy(rows)
    validate_split_groups({"train": rows})
    assert rows == original
