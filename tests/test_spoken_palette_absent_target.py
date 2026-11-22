from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from scripts.pretrained_reuse.audit_spoken_palette_absent_target import (
    _group_train_sources,
    binary_auc,
    replace_palette_color,
    score_summary,
)
from scripts.spoken_palette_decision_data import RGB


def test_replace_palette_color_changes_only_the_requested_palette_fill() -> None:
    pixels = np.zeros((40, 40, 3), dtype=np.uint8)
    pixels[5:25, 5:25] = RGB["blue"]
    pixels[25:35, 25:35] = (123, 45, 67)
    image = Image.fromarray(pixels, mode="RGB")

    replaced, count = replace_palette_color(image, removed_color="blue", replacement_color="red")

    result = np.asarray(replaced)
    assert count == 400
    assert np.count_nonzero(np.all(result == RGB["blue"], axis=-1)) == 0
    assert np.count_nonzero(np.all(result == RGB["red"], axis=-1)) == 400
    assert tuple(result[30, 30]) == (123, 45, 67)


def test_replace_palette_color_rejects_missing_fill_and_same_color() -> None:
    with pytest.raises(ValueError, match="fill area"):
        replace_palette_color(
            Image.new("RGB", (20, 20), "black"), removed_color="blue", replacement_color="red"
        )
    with pytest.raises(ValueError, match="two distinct colors"):
        replace_palette_color(
            Image.new("RGB", (20, 20), RGB["blue"]), removed_color="blue", replacement_color="blue"
        )


def test_binary_auc_handles_perfect_and_tied_scores() -> None:
    assert binary_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert binary_auc([0, 1], [0.5, 0.5]) == 0.5


def test_train_source_groups_follow_frozen_corpus_ids_not_all_official_train_rows() -> None:
    source_rows = {
        f"{speaker}-{color}": {
            "source_id": f"{speaker}-{color}",
            "speaker_group": speaker,
            "targets": {"color": color},
        }
        for speaker in ("train-a", "train-b", "held-out")
        for color in RGB
    }
    corpus_ids = {
        f"{speaker}-{color}" for speaker in ("train-a", "train-b") for color in RGB
    }

    selected, groups = _group_train_sources(source_rows, corpus_ids)

    assert len(selected) == 8
    assert set(groups) == {"train-a", "train-b"}
    assert all(colors == set(RGB) for colors in groups.values())


def test_train_source_groups_reject_unknown_frozen_corpus_ids() -> None:
    with pytest.raises(ValueError, match="absent from the official TRAIN layout"):
        _group_train_sources({}, {"missing-source"})


def test_score_summary_keeps_invalid_target_as_diagnostic_not_abstention() -> None:
    rows = []
    for sample, group, invalid_score in (
        ("s1", "g1", 0.8),
        ("s2", "g2", 0.2),
    ):
        for condition, uncertainty in (
            ("matched", 0.1),
            ("target_absent", invalid_score),
            ("distractor_absent", 0.15),
        ):
            rows.append(
                {
                    "sample_id": sample,
                    "speaker_group": group,
                    "condition": condition,
                    "max_probability": 1.0 - uncertainty,
                    "top_two_margin": max(0.0, 1.0 - 2.0 * uncertainty),
                    "uncertainty_score": uncertainty,
                    "uncertainty_margin_score": uncertainty,
                    "normalized_entropy": uncertainty,
                    "prediction_matches_target_position": condition != "target_absent",
                }
            )

    result = score_summary(rows)
    assert (
        result["target_absence_detection"]["pooled_auc_uncertainty_1_minus_max_probability"] == 1.0
    )
    assert result["conditions"]["matched"]["target_present_accuracy"] == 1.0
    assert result["conditions"]["target_absent"]["target_present_accuracy"] is None
    assert result["runtime_abstention_supported"] is False
    assert result["target_absence_detection"]["threshold_selected"] is False


def test_binary_auc_rejects_single_class_or_nonfinite_scores() -> None:
    with pytest.raises(ValueError, match="both classes"):
        binary_auc([1, 1], [0.1, 0.2])
    with pytest.raises(ValueError, match="both classes"):
        binary_auc([0, 1], [0.1, float("nan")])
