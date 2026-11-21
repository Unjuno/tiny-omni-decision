from __future__ import annotations

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.train_habit_temporal_decision import (  # noqa: E402
    _batch,
    _group_bootstrap_deltas,
    _metrics,
    _new_model,
    _video_derangement,
)


def test_metrics_report_sample_and_equal_group_scores() -> None:
    logits = [torch.tensor([4.0, 0.0]), torch.tensor([0.0, 4.0]), torch.tensor([0.0, 4.0])]

    result = _metrics(logits, [0, 0, 1], ["g1", "g1", "g2"])

    assert result["count"] == 3
    assert result["group_count"] == 2
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["macro_group_accuracy"] == pytest.approx(0.75)
    assert result["probabilities"][0][0] > result["probabilities"][0][1]


def test_metrics_require_aligned_nonempty_rows() -> None:
    with pytest.raises(ValueError, match="aligned nonempty"):
        _metrics([], [], [])
    with pytest.raises(ValueError, match="aligned nonempty"):
        _metrics([torch.tensor([1.0, 0.0])], [0], [])


def test_group_bootstrap_is_paired_deterministic_and_labels_selection_effect() -> None:
    arms = {}
    for name, offset in (("video_state", 0.1), ("text_only", 0.0),
                         ("video_only", 0.02), ("state_only", -0.01)):
        arms[name] = {"validation": {"per_group": {
            f"g{index}": {
                "accuracy": 0.5 + offset,
                "nll": 1.0 - offset,
            }
            for index in range(6)
        }}}

    result = _group_bootstrap_deltas(arms, draws=200)
    repeated = _group_bootstrap_deltas(arms, draws=200)

    assert result == repeated
    assert result["draws"] == 200
    assert result["selection_affected_validation"] is True
    assert result["comparisons"]["video_state_minus_text_only"]["accuracy"][
        "mean_group_delta"
    ] == pytest.approx(0.1)


def test_video_derangement_matches_imbalanced_groups_deterministically() -> None:
    examples = [
        {"sid": group}
        for group, count in (("large", 28), ("b", 12), ("c", 12),
                             ("d", 8), ("e", 8), ("f", 8))
    ]
    indices = list(range(len(examples)))

    mapping = _video_derangement(indices, examples, seed=17)
    repeated = _video_derangement(indices, examples, seed=17)

    donors = [mapping[index] for index in indices]
    assert mapping == repeated
    assert sorted(donors) == indices
    assert all(examples[index]["sid"] != examples[mapping[index]]["sid"]
               for index in indices)


def test_video_derangement_rejects_impossible_group_skew() -> None:
    examples = [
        {"sid": group}
        for group, count in (("large", 6), ("small", 2))
        for _ in range(count)
    ]

    with pytest.raises(ValueError, match="derangement is impossible"):
        _video_derangement(list(range(len(examples))), examples)


def test_candidate_order_randomization_keeps_target_bound_to_candidate() -> None:
    examples = [{
        "id": "train:1:0", "sid": "train-group", "split": "train",
        "target_index": 1, "options": ["left", "right"],
        "context_end_ms": 700.0,
        "video_frame_times_ms": [0, 100, 200, 300, 400, 500, 600, 700],
        "sensor_frame_times_ms": [0, 100, 200, 300, 400, 500, 600, 700],
    }]
    arrays = {
        "video_features": np.ones((1, 8, 768), dtype=np.float32),
        "sensor_sequence": np.ones((1, 8, 14), dtype=np.float32),
    }
    context = torch.zeros((1, 384))
    option_rows = [torch.stack([torch.ones(1536), torch.full((1536,), 2.0)])]
    batch = _batch(
        [0], examples, arrays, context, option_rows, torch.zeros(14), torch.ones(14),
        "video_state", shuffle_options=True, rng=random.Random(17),
    )

    assert batch[8].shape == (1, 2, 1536)
    selected_candidate = batch[8][0, batch[11][0].item()]
    assert selected_candidate.eq(2.0).all()
    assert batch[0].shape == (1, 8, 768)
    assert batch[4].shape == (1, 8, 14)
    assert batch[1][0, -1].item() == pytest.approx(0.0)


def test_all_multimodal_training_arms_have_finite_fixed_size() -> None:
    for mode in ("video_only", "state_only", "video_state"):
        model = _new_model(mode)
        assert sum(parameter.numel() for parameter in model.parameters()) > 0
        assert all(parameter.requires_grad for parameter in model.parameters())

