import pytest

from scripts.evaluate_coverage_transfer import validate_transfer_disjoint


def test_transfer_rejects_latent_or_state_overlap():
    train = [dict(id="train", state='{"x":1}', group="a")]
    validate_transfer_disjoint(train, [dict(id="eval", state='{"x":2}', group="b")])
    with pytest.raises(ValueError):
        validate_transfer_disjoint(train, [dict(id="eval", state='{"x":2}', group="a")])
    with pytest.raises(ValueError):
        validate_transfer_disjoint(train, [dict(id="eval", state='{"x":1}', group="b")])
