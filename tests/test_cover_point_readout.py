import numpy as np
import pytest

torch = pytest.importorskip("torch")
from tiny_omni_decision.cover_point_readout import (  # noqa: E402
    CoverPointReadout,
    balanced_membership_loss,
    initial_cover_membership,
)


def source(box=(0.2, 0.2, 0.4, 0.4)):
    return {
        "grounded_question": [
            {"question": "Track the object that occludes the smaller object.", "answers": [4]}
        ],
        "object_tracking": [{"id": 4, "frame_ids": [0, 10], "bounding_boxes": [box, box]}],
    }


def test_supervision_only_preserves_complete_grid_and_boundaries():
    xy = np.array([[0.2, 0.2], [0.4, 0.4], [0.5, 0.3]])
    np.testing.assert_array_equal(initial_cover_membership(source(), xy), [True, True, False])
    assert not initial_cover_membership(source((0.01, 0.01, 0.02, 0.02)), xy).any()
    np.testing.assert_array_equal(xy, [[0.2, 0.2], [0.4, 0.4], [0.5, 0.3]])


@pytest.mark.parametrize("kind", ["missing", "duplicate", "track", "frame", "box", "nonfinite"])
def test_missing_or_invalid_supervision_fails_closed(kind):
    row = source()
    if kind == "missing":
        row["grounded_question"] = []
    elif kind == "duplicate":
        row["grounded_question"] *= 2
    elif kind == "track":
        row["grounded_question"][0]["answers"] = [99]
    elif kind == "frame":
        row["object_tracking"][0]["frame_ids"] = [1, 10]
    elif kind == "box":
        row["object_tracking"][0]["bounding_boxes"][0] = [0.5, 0.2, 0.4, 0.4]
    else:
        row["object_tracking"][0]["bounding_boxes"][0] = [float("nan"), 0.2, 0.4, 0.4]
    with pytest.raises(ValueError):
        initial_cover_membership(row, np.array([[0.3, 0.3]]))


def test_point_equivariance_padding_and_gradient():
    torch.manual_seed(17)
    model = CoverPointReadout(input_dim=5, hidden=7)
    x = torch.randn(2, 4, 5)
    logits = model(x)
    torch.testing.assert_close(model(x.flip(1)), logits.flip(1))
    pad = torch.cat([x, torch.full((2, 3, 5), 99.0)], 1)
    mask = torch.tensor([[True] * 4 + [False] * 3] * 2)
    torch.testing.assert_close(model(pad, mask)[:, :4], logits)
    assert torch.isneginf(model(pad, mask)[:, 4:]).all()
    loss = balanced_membership_loss(
        logits, torch.tensor([[True, False, False, False], [False] * 4])
    )
    loss.backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())
    assert any(p.grad.abs().sum() > 0 for p in model.parameters())
    restored = CoverPointReadout(input_dim=5, hidden=7)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(x), logits, rtol=0, atol=0)


def test_balanced_loss_counts_clips_and_retains_no_positive_rows():
    logits = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    target = torch.tensor([[True, False, False], [False, False, False]])
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target.float(), reduction="none"
    )
    expected = (0.5 * (bce[0, 0] + bce[0, 1:].mean()) + bce[1].mean()) / 2
    torch.testing.assert_close(balanced_membership_loss(logits, target), expected)
    torch.testing.assert_close(
        balanced_membership_loss(logits, ~target), balanced_membership_loss(-logits, target)
    )


def test_model_and_loss_reject_invalid_inputs():
    model = CoverPointReadout(input_dim=5)
    for x in [torch.zeros(1, 0, 5), torch.zeros(1, 2, 4), torch.full((1, 2, 5), float("nan"))]:
        with pytest.raises(ValueError):
            model(x)
    with pytest.raises(ValueError):
        model(torch.zeros(1, 2, 5), torch.zeros(1, 2, dtype=torch.bool))
    with pytest.raises(ValueError):
        balanced_membership_loss(torch.zeros(1, 2), torch.zeros(1, 2))
    with pytest.raises(ValueError):
        balanced_membership_loss(
            torch.full((1, 2), float("inf")), torch.zeros(1, 2, dtype=torch.bool)
        )
