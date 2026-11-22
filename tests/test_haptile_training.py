from __future__ import annotations

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.pretrained_reuse.train_haptile_decision import (  # noqa: E402
    _batch_tensors,
    _metrics,
)


def test_metrics_report_sample_and_task_macro_scores_and_each_prediction():
    logits = torch.tensor([
        [6.0, 0.0, 0, 0, 0, 0, 0],
        [0.0, 6.0, 0, 0, 0, 0, 0],
        [0.0, 6.0, 0, 0, 0, 0, 0],
    ])
    targets = torch.tensor([0, 1, 0])
    result = _metrics(logits, targets, ["task_a", "task_a", "task_b"], ["a", "b", "c"])
    assert result["count"] == 3
    assert result["task_group_count"] == 2
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["macro_task_group"]["accuracy"] == pytest.approx(0.5)
    assert result["macro_task_group_nll"] > 0
    assert [row["sample_id"] for row in result["predictions"]] == ["a", "b", "c"]
    assert result["predictions"][0]["target"] == "-x"
    assert set(result["predictions"][0]["probabilities"]) == {
        "-x", "+x", "-y", "+y", "-z", "+z", "hold"
    }


def test_option_shuffle_keeps_target_attached_to_its_option():
    example_count = 3
    options = np.stack([
        np.tile(np.arange(7, dtype=np.float32)[:, None], (1, 1536)) + 10 * row
        for row in range(example_count)
    ])
    arrays = {
        "targets": np.asarray([0, 4, 6], dtype=np.int64),
        "option_features": options,
        "option_features_no_instruction": options + 100,
        "context_features": np.zeros((example_count, 384), dtype=np.float32),
        "context_features_no_instruction": np.ones((example_count, 384), dtype=np.float32),
        "video_values": np.zeros((example_count, 4, 8, 768), dtype=np.float32),
        "video_timestamps_ms": np.tile(np.arange(8, dtype=np.float32), (example_count, 1)),
        "state_values": np.zeros((example_count, 8, 25), dtype=np.float32),
        "state_timestamps_ms": np.tile(np.arange(8, dtype=np.float32), (example_count, 1)),
    }
    batch = _batch_tensors(
        arrays, [0, 1, 2], torch.zeros(25), torch.ones(25),
        mode="text_only", shuffle_options=True, rng=random.Random(17),
    )
    _, _, _, _, _, options_in_batch, _, targets = batch
    for row, target in enumerate(targets.tolist()):
        assert options_in_batch[row, target, 0].item() == options[row, [0, 4, 6][row], 0]


def test_state_only_batch_uses_instruction_free_features():
    arrays = {
        "targets": np.asarray([3], dtype=np.int64),
        "option_features": np.zeros((1, 7, 1536), dtype=np.float32),
        "option_features_no_instruction": np.ones((1, 7, 1536), dtype=np.float32),
        "context_features": np.zeros((1, 384), dtype=np.float32),
        "context_features_no_instruction": np.ones((1, 384), dtype=np.float32),
        "video_values": np.zeros((1, 4, 8, 768), dtype=np.float32),
        "video_timestamps_ms": np.tile(np.arange(8, dtype=np.float32), (1, 1)),
        "state_values": np.zeros((1, 8, 25), dtype=np.float32),
        "state_timestamps_ms": np.tile(np.arange(8, dtype=np.float32), (1, 1)),
    }
    batch = _batch_tensors(
        arrays, [0], torch.zeros(25), torch.ones(25), mode="state_only",
        shuffle_options=False, rng=random.Random(17),
    )
    assert torch.equal(batch[4], torch.ones((1, 384)))
    assert torch.equal(batch[5], torch.ones((1, 7, 1536)))
