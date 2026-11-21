import numpy as np
import pytest

from scripts.audit_r2m_temporal_bandwidth import select_train_rows
from tiny_omni_decision.r2m_temporal_bandwidth import (
    effective_uniform_sample_rate,
    out_of_band_energy_fraction,
)


def row(identifier, activity, participant, split="train"):
    return {
        "sample_id": identifier,
        "activity": activity,
        "participant_id": participant,
        "trial_id": int(identifier.rsplit("=", 1)[1]),
        "split": split,
    }


def test_effective_rate_accounts_for_endpoint_inclusive_grid():
    assert effective_uniform_sample_rate(128, 2.0) == pytest.approx(63.5)


def test_out_of_band_energy_reports_above_target_nyquist():
    source_rate = 1000.0
    time = np.arange(1000) / source_rate
    signal = np.sin(2 * np.pi * 100 * time) + 0.1 * np.sin(2 * np.pi * 10 * time)
    assert out_of_band_energy_fraction(signal, source_rate, 50.0) > 0.98
    assert out_of_band_energy_fraction(signal, source_rate, 500.0) < 0.02


def test_train_sample_selection_is_deterministic_and_split_safe():
    rows = [
        row("a|participant=2|trial=1", "walk", 2),
        row("a|participant=2|trial=2", "walk", 2),
        row("a|participant=4|trial=3", "walk", 4),
        row("b|participant=12|trial=4", "walk", 12, "calibration"),
    ]
    first = select_train_rows(rows)
    second = select_train_rows(list(reversed(rows)))
    assert [item["sample_id"] for item in first] == [item["sample_id"] for item in second]
    assert len(first) == 2
    assert {item["participant_id"] for item in first} == {2, 4}


def test_train_sample_selection_rejects_unpinned_participant_ids():
    with pytest.raises(ValueError, match="outside split v2"):
        select_train_rows([row("a|participant=12|trial=1", "walk", 12)])
