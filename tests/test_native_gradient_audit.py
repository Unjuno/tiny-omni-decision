import ast
import hashlib
import random
from pathlib import Path

import pytest
import torch

from scripts.native_gradient_audit import (
    SAMPLED_STEPS,
    cosine_and_norms,
    flatten_gradients,
    option_order,
    selected_updates,
    sha_file,
)


def test_sampled_updates_are_evenly_stratified_and_pin_exact_sampler_rows():
    order = [[step * 8 + index for index in range(8)] for step in range(2048)]
    occurrences = [[step * 8 + index for index in range(8)] for step in range(2048)]
    selected = selected_updates(order, occurrences)

    assert len(selected) == 32
    assert selected[0]["optimizer_step"] == 1
    assert selected[-1]["optimizer_step"] == 2048
    assert [row["optimizer_step"] - 1 for row in selected] == list(SAMPLED_STEPS)
    assert all(len(row["row_indices"]) == len(row["occurrence_ids"]) == 8 for row in selected)


@pytest.mark.parametrize(
    ("order", "occurrences"),
    [([], []), ([[0] * 8] * 2048, [[0] * 8] * 2048)],
)
def test_sampled_updates_reject_incomplete_or_ambiguous_sampler(order, occurrences):
    with pytest.raises(ValueError):
        selected_updates(order, occurrences)


def test_gradient_summary_exposes_opposition_clipping_and_cancellation():
    gradients = {
        "text": torch.tensor([1.0, 0.0]),
        "image": torch.tensor([-1.0, 0.0]),
        "audio": torch.tensor([0.0, 2.0]),
        "video": torch.tensor([0.0, -2.0]),
    }

    result = cosine_and_norms(gradients)

    assert result["pairwise_cosines"]["text|image"] == -1.0
    assert result["pairwise_cosines"]["audio|video"] == -1.0
    assert result["cancellation_ratio"] == 0.0
    assert result["unclipped_mean_batch_gradient_norm"] == 0.0
    assert result["clip_factor_at_training_max_norm_1"] == 1.0
    assert result["modality_projection_shares"] == {
        "text": None,
        "image": None,
        "audio": None,
        "video": None,
    }


def test_gradient_summary_reports_actual_global_clip_factor():
    gradients = {name: torch.tensor([4.0, 0.0]) for name in ("text", "image", "audio", "video")}
    result = cosine_and_norms(gradients)

    assert result["unclipped_mean_batch_gradient_norm"] == 4.0
    assert result["clip_factor_at_training_max_norm_1"] == 0.25
    assert result["would_clip"] is True


def test_unused_modality_parameters_contribute_exact_zero_gradients():
    used = torch.tensor([1.0, 2.0], requires_grad=True)
    unused = torch.tensor([3.0, 4.0], requires_grad=True)
    gradients = torch.autograd.grad(used.square().sum(), (used, unused), allow_unused=True)

    assert gradients[1] is None
    assert torch.equal(
        flatten_gradients(gradients, (used, unused)), torch.tensor([2.0, 4.0, 0.0, 0.0])
    )


def test_option_order_is_deterministic_and_is_a_permutation():
    row = {"id": "fixed-train-row", "options": ["a", "b", "c", "d"]}

    first = option_order(row, 123)

    assert first == option_order(row, 123)
    assert sorted(first) == [0, 1, 2, 3]
    assert first != option_order(row, 124)


def test_option_order_matches_the_pinned_run_helper():
    root = (
        Path(__file__).parents[1]
        / "docs/pretrained_reuse/source_snapshots/native_coverage_repeat_fit_01/scripts"
    )
    tree = ast.parse((root / "native_fit.py").read_text(encoding="utf-8"))
    node = next(
        value
        for value in tree.body
        if isinstance(value, ast.FunctionDef) and value.name == "option_order"
    )
    namespace = {"hashlib": hashlib, "random": random}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "pinned-option-order", "exec"), namespace
    )
    row = {"id": "fixed-train-row", "options": ["a", "b", "c", "d", "e"]}

    for occurrence in (0, 1, 127, 2047, 15872):
        assert option_order(row, occurrence) == namespace["option_order"](row, occurrence)


def test_bundled_source_snapshot_matches_registered_hashes():
    from pathlib import Path

    root = (
        Path(__file__).parents[1]
        / "docs/pretrained_reuse/source_snapshots/native_coverage_repeat_fit_01"
    )
    expected = {
        "scripts/native_fit.py": "62315a27e37c9b3c2837ee66c2d462c4285b4aca1a739fb769974c3330f8951b",
        "scripts/native_fit_runtime.py": (
            "63598df78a632b6a2a49c4d70f9c9e07a945f94793e07cb4144ec2234f346abe"
        ),
        "scripts/native_candidate_inputs.py": (
            "372e8c3664fbfa8627a9bba4ff4c7ab3c0f3fc92d8a0c5df3e8011b1a45cd7f3"
        ),
        "scripts/native_video_cache.py": (
            "62ffe34c6db4a52ed93310c36939a74f356729db338a7aea49406f1cdfe58039"
        ),
        "tiny_omni_decision/native_candidate.py": (
            "cd2ab4e45de5c5e3e037531bf8a9e967365f8ef2a60b5b20cddd0ba3e336a974"
        ),
    }

    assert {name: sha_file(root / name) for name in expected} == expected
