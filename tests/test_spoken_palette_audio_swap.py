from __future__ import annotations

import pytest

from scripts.pretrained_reuse.audit_spoken_palette_audio_swap import (
    make_audio_swap_plan,
    summarize,
)
from scripts.spoken_palette_decision_data import COLORS, POSITIONS


def _pairs() -> list[dict[str, object]]:
    rows = []
    for group in (f"speaker-{index:02d}" for index in range(24)):
        for color_index, color in enumerate(COLORS):
            layout = list(COLORS[color_index:] + COLORS[:color_index])
            position_index = layout.index(color)
            rows.append(
                {
                    "sample_id": f"{group}-{color}",
                    "source_id": f"{group}-{color}",
                    "speaker_group": group,
                    "target_color": color,
                    "target_position": list(POSITIONS[position_index]),
                    "target_position_index": position_index,
                    "layout_colors_by_position": list(layout),
                    "options": [
                        {
                            "id": f"option-{position_index_for_option}",
                            "x_norm": POSITIONS[position_index_for_option][0],
                            "y_norm": POSITIONS[position_index_for_option][1],
                        }
                        for position_index_for_option in range(len(POSITIONS))
                    ],
                }
            )
    return rows


def test_audio_rotation_keeps_screen_fixed_and_maps_spoken_color_to_option() -> None:
    plan = make_audio_swap_plan(_pairs())

    assert len(plan) == 96
    assert all(row["screen_color"] != row["spoken_color"] for row in plan)
    assert all(row["target_position_changed"] is True for row in plan)
    assert len({row["audio_pair"]["source_id"] for row in plan}) == 96


def test_audio_rotation_rejects_missing_speaker_color_source() -> None:
    pairs = _pairs()
    pairs = [
        row
        for row in pairs
        if not (row["speaker_group"] == "speaker-00" and row["target_color"] == COLORS[1])
    ]

    with pytest.raises(ValueError, match="speaker/color pair missing"):
        make_audio_swap_plan(pairs)


def test_audio_swap_summary_bootstraps_whole_speaker_groups() -> None:
    records = [
        {
            "speaker_group": f"speaker-{group:02d}",
            "prediction_matches_spoken_color": sample_index % 2 == 0,
            "target_position_changed": True,
            "target_option_probability": 0.75,
            "prediction_changed_from_matched": sample_index % 2 == 0,
        }
        for group in range(24)
        for sample_index in range(4)
    ]

    result = summarize(records, replicates=200, seed=17)

    assert result["independent_groups"] == 24
    assert result["source_pairs"] == 96
    assert result["metrics"]["rotated_audio_target_accuracy"]["estimate"] == 0.5
    assert result["metrics"]["changed_position_accuracy"]["estimate"] == 0.5
    assert result["metrics"]["mean_prediction_change_from_matched"]["estimate"] == 0.5
