from __future__ import annotations

import pytest

from scripts.evaluate_clevr_cogent_b_audit import paired_bootstrap


def _row(sample_id: str, target: int, probabilities: list[float]) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "target_position": target,
        "option_probabilities": probabilities,
    }


def test_paired_bootstrap_handles_variable_option_counts_deterministically() -> None:
    real = [
        _row("image-1", 0, [0.8, 0.2]),
        _row("image-2", 2, [0.2, 0.3, 0.5]),
        _row("image-3", 1, [0.2, 0.8]),
    ]
    control = [
        _row("image-1", 0, [0.4, 0.6]),
        _row("image-2", 2, [0.5, 0.25, 0.25]),
        _row("image-3", 1, [0.6, 0.4]),
    ]
    first = paired_bootstrap(real, control, replicates=100, seed=17)
    second = paired_bootstrap(real, control, replicates=100, seed=17)

    assert first == second
    assert first["unit_count"] == 3
    assert first["interval_95_percent"]["accuracy_delta"] == [1.0, 1.0]


def test_paired_bootstrap_rejects_misaligned_sample_ids() -> None:
    with pytest.raises(ValueError, match="identical sample ID order"):
        paired_bootstrap(
            [_row("image-1", 0, [0.8, 0.2])],
            [_row("different-image", 0, [0.4, 0.6])],
            replicates=10,
        )
