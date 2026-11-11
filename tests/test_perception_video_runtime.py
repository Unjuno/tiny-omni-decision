from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tiny_omni_decision.perception_video_readout import PerceptionVideoDecisionReadout  # noqa: E402
from tiny_omni_decision.perception_video_runtime import (  # noqa: E402
    READOUT_CONFIG_SHA256,
    READOUT_SEED,
    READOUT_SELECTED_EPOCH,
    _candidate_text,
    load_readout_checkpoint,
    pool_candidate_queries,
    uniform_frame_indices,
)


def test_perception_video_readout_is_candidate_permutation_equivariant() -> None:
    torch.manual_seed(3)
    model = PerceptionVideoDecisionReadout().eval()
    video = torch.randn(1, 9, 384)
    queries = torch.randn(1, 3, 384)

    with torch.inference_mode():
        original = model(video, queries)
        permuted = model(video, queries[:, [2, 0, 1]])

    assert model.parameter_count() == 444_673
    torch.testing.assert_close(permuted, original[:, [2, 0, 1]], rtol=0, atol=1e-6)


def test_video_frame_indices_are_endpoint_inclusive_and_deterministic() -> None:
    indices = uniform_frame_indices(100)

    assert indices == uniform_frame_indices(100)
    assert len(indices) == 16
    assert indices[0] == 0
    assert indices[-1] == 99
    assert indices == sorted(set(indices))


def test_video_runtime_candidate_text_strips_only_training_prefixes() -> None:
    assert _candidate_text(" A: red cube ") == "red cube"
    assert _candidate_text("no prefix") == "no prefix"

    with pytest.raises(ValueError, match="nonempty"):
        _candidate_text("C:   ")


def test_pooled_candidate_queries_include_reader_batch_axis() -> None:
    hidden = torch.ones(3, 5, 384)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])

    queries = pool_candidate_queries(hidden, mask)

    assert tuple(queries.shape) == (1, 3, 384)
    torch.testing.assert_close(queries, torch.ones_like(queries), rtol=0, atol=0)


def test_readout_checkpoint_loader_accepts_pinned_training_envelope(tmp_path: Path) -> None:
    torch.manual_seed(17)
    source = PerceptionVideoDecisionReadout().eval()
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "state_dict": source.state_dict(),
            "selected_epoch": READOUT_SELECTED_EPOCH,
            "seed": READOUT_SEED,
            "config_sha256": READOUT_CONFIG_SHA256,
        },
        path,
    )
    target = PerceptionVideoDecisionReadout().eval()

    load_readout_checkpoint(target, path)

    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(
            source.state_dict().values(), target.state_dict().values(), strict=True
        )
    )


def test_readout_checkpoint_loader_rejects_unpinned_run_identity(tmp_path: Path) -> None:
    model = PerceptionVideoDecisionReadout().eval()
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "selected_epoch": READOUT_SELECTED_EPOCH,
            "seed": 999,
            "config_sha256": READOUT_CONFIG_SHA256,
        },
        path,
    )

    with pytest.raises(ValueError, match="identity"):
        load_readout_checkpoint(model, path)
