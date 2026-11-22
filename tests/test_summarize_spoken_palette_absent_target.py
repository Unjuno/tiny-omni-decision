from __future__ import annotations

import pytest

from scripts.pretrained_reuse.summarize_spoken_palette_absent_target import summarize


def _paired_rows() -> list[dict[str, object]]:
    rows = []
    for group in ("speaker-a", "speaker-b"):
        for sample_index in range(2):
            sample = f"{group}-{sample_index}"
            for condition, confidence, matches in (
                ("matched", 0.95, True),
                ("target_absent", 0.55, False),
                ("distractor_absent", 0.94, True),
            ):
                rows.append(
                    {
                        "speaker_group": group,
                        "sample_id": sample,
                        "condition": condition,
                        "max_probability": confidence,
                        "uncertainty_score": 1.0 - confidence,
                        "prediction_matches_target_position": matches,
                    }
                )
    return rows


def test_group_bootstrap_is_deterministic_and_preserves_paired_conditions() -> None:
    first = summarize(_paired_rows(), replicates=200, seed=17)
    second = summarize(_paired_rows(), replicates=200, seed=17)

    assert first == second
    assert first["independent_groups"] == 2
    assert first["source_pairs"] == 4
    assert first["metrics"]["target_absence_uncertainty_auc"]["estimate"] == 1.0
    assert first["metrics"]["target_absent_max_probability_delta"]["estimate"] == pytest.approx(
        -0.4
    )


def test_group_bootstrap_rejects_unpaired_sources() -> None:
    rows = _paired_rows()[:-1]
    with pytest.raises(ValueError, match="all three paired conditions"):
        summarize(rows, replicates=200)


def test_group_bootstrap_requires_enough_replicates() -> None:
    with pytest.raises(ValueError, match="at least 100"):
        summarize(_paired_rows(), replicates=99)
