import numpy as np
import pytest
import torch

from scripts.diagnose_native_weak_input_use import (
    choose_donors,
    group_bootstrap_deltas,
    replace_observation,
    zero_blocks,
)
from tiny_omni_decision.native_candidate import NativeTokenBlock


def _row(ident, group, *, task="color", modality="image", options=2):
    return {
        "id": ident,
        "group": group,
        "modality": modality,
        "source": "fixture",
        "task": task,
        "observation": "obs-" + ident,
        "options": list(range(options)),
        "target_index": 0,
    }


def test_choose_donors_keeps_task_profile_and_changes_group():
    rows = [_row("a", "g1"), _row("b", "g1"), _row("c", "g2"), _row("d", "g3")]
    profiles = {row["observation"]: ("image",) for row in rows}
    donors = choose_donors(rows, profiles)
    assert set(donors) == {row["id"] for row in rows}
    assert all(donors[row["id"]]["group"] != row["group"] for row in rows)
    assert all(donors[row["id"]]["task"] == row["task"] for row in rows)


def test_choose_donors_leaves_single_group_bucket_unmatched():
    rows = [_row("a", "g1"), _row("b", "g1")]
    assert choose_donors(rows) == {}


def test_zero_blocks_preserves_metadata_without_mutating_input():
    values = torch.ones(3, 4)
    positions = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    valid = torch.tensor([True, False, True])
    original = NativeTokenBlock("text", values, positions, valid)

    result = zero_blocks([original])[0]

    assert torch.equal(result.values, torch.zeros_like(values))
    assert torch.equal(original.values, torch.ones_like(values))
    assert result.positions is positions
    assert result.valid is valid
    assert result.kind == "text"


def test_replace_observation_accepts_variable_token_length():
    current = NativeTokenBlock(
        "text", torch.ones(2, 4), torch.zeros(2, 3), torch.ones(2, dtype=torch.bool)
    )
    donor_positions = torch.ones(3, 3)
    donor_mask = torch.tensor([True, True, False])
    donor = NativeTokenBlock("text", torch.full((3, 4), 2.0), donor_positions, donor_mask)

    result = replace_observation([current], [donor])[0]

    assert result.values.shape == (3, 4)
    assert torch.equal(result.values, donor.values)
    assert result.positions is donor_positions
    assert result.valid is donor_mask


def test_group_bootstrap_is_paired_and_deterministic():
    rows = [_row("a", "g1"), _row("b", "g1"), _row("c", "g2"), _row("d", "g3")]
    baseline = np.asarray([[4, 0], [0, 4], [4, 0], [0, 4]], dtype=np.float32)
    variant = np.asarray([[4, 0], [4, 0], [4, 0], [4, 0]], dtype=np.float32)
    first = group_bootstrap_deltas(rows, baseline, variant, draws=300)
    second = group_bootstrap_deltas(rows, baseline, variant, draws=300)
    assert first == second
    assert first["accuracy"]["point_delta"] == pytest.approx(0.5)
    assert first["nll"]["point_delta"] < 0
    assert len(first["brier"]["ci95_percentile"]) == 2
